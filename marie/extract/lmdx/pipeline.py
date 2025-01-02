import os
from collections import Counter

import pandas as pd
import pytesseract

from .utils import format_segment, normalize_box, regex_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Ref https://arxiv.org/pdf/2309.10952


class Lmdx:
    def __init__(self, file_path, tokenizer):
        self.images = self.image_from_pdf(file_path)
        self.chunks = self.create_chunks(self.images, tokenizer)

    def generate_prompt(self, schema):
        """
        Create a list of prompts for N document chunks for a given schema
        """
        prompts = []
        for chunk in self.chunks:
            document = ""
            for index, row in chunk.iterrows():
                document = (
                    document + str(row["segment"]) + " " + row["segment_id"] + "\n"
                )
            prompt = f"<Document>{document}</Document><Task>From the document, extract the text values and tags\
                     of the following entities:{schema}</task><Extraction>"
            prompts.append(prompt)
        return prompts

    def _quantize(self, values, n=99):
        """
        Takes a list of values and returns the returns a list of buckets
        after quantizing the values to n buckets
        """
        min_value = min(values)
        max_value = max(values)
        bucket_width = (max_value - min_value) / n
        bucket_indices = []
        for value in values:
            index = int(
                (value - min_value) / bucket_width
            )  # Determine which bucket it belongs to
            bucket_indices.append(index)
        return bucket_indices

    def _apply_ocr(self, image):
        """
        Applies OCR on an image, and
        returns words, blocks and normalized X,Y centres.
        """
        data = pytesseract.image_to_data(image, output_type="data.frame").dropna()
        words, left, top, width, height = (
            data["text"].astype(str),
            data["left"],
            data["top"],
            data["width"],
            data["height"],
        )
        data["line_block"] = data.apply(
            lambda row: [row["block_num"], row["line_num"]], axis=1
        ).astype(str)

        # filter empty words and corresponding coordinates
        irrelevant_indices = [idx for idx, word in enumerate(words) if not word.strip()]
        words = [
            word for idx, word in enumerate(words) if idx not in irrelevant_indices
        ]
        line_blocks = [
            coord
            for idx, coord in enumerate(data["line_block"])
            if idx not in irrelevant_indices
        ]
        left = [
            coord for idx, coord in enumerate(left) if idx not in irrelevant_indices
        ]
        top = [coord for idx, coord in enumerate(top) if idx not in irrelevant_indices]
        width = [
            coord for idx, coord in enumerate(width) if idx not in irrelevant_indices
        ]
        height = [
            coord for idx, coord in enumerate(height) if idx not in irrelevant_indices
        ]

        # turn coordinates into (left, top, left+width, top+height) format
        actual_boxes = []
        for x, y, w, h in zip(left, top, width, height):
            actual_box = [x, y, x + w, y + h]
            actual_boxes.append(actual_box)

        image_width, image_height = image.size

        # finally, normalize the bounding boxes
        normalized_boxes = []
        for box in actual_boxes:
            normalized_box = normalize_box(box, image_width, image_height)
            Xmid = (normalized_box[0] + normalized_box[2]) / 2
            Ymid = (normalized_box[1] + normalized_box[3]) / 2
            normalized_boxes.append([Xmid, Ymid])

        if len(words) != len(normalized_boxes):
            raise ValueError("Not as many words as there are bounding boxes")

        return words, line_blocks, normalized_boxes

    def _create_segments(self, image):
        """
        Create segments and segment ids from a document chunk.
        """
        words, lines, normalized_boxes = self._apply_ocr(image)

        data = pd.DataFrame(
            {
                "words": words,
                "lines": lines,
                "Xmid": [box[0] for box in normalized_boxes],
                "Ymid": [box[1] for box in normalized_boxes],
            }
        )

        data["segment"] = data.groupby("lines")["words"].transform(
            lambda x: " ".join(map(str, x))
        )

        data["x-mid"] = data.groupby("lines")["Xmid"].transform("mean")
        data["y-mid"] = data.groupby("lines")["Ymid"].transform("mean")

        # Drop unnecessary columns and duplicates
        data = data[["segment", "x-mid", "y-mid"]].drop_duplicates(keep="first")

        # Apply quantize function to 'x-mid' and 'y-mid'
        data["x-mid"] = self._quantize(data["x-mid"])
        data["y-mid"] = self._quantize(data["y-mid"])

        data["segment_id"] = data.apply(format_segment, axis=1)

        return data

    def create_chunks(self, images, tokenizer):
        """
        Divides a document into chunks based on max token length.
        """
        chunks = []
        max_token_limit = tokenizer.model_max_length

        for image in images:
            token_count = 0
            df = self._create_segments(image)
            df_list = []
            current_df = pd.DataFrame(columns=df.columns)

            # Iterate through the DataFrame
            for index, row in df.iterrows():
                text = row["segment"]
                tokens = tokenizer.encode(text, add_special_tokens=True)

                # Check if adding this row's tokens would exceed the limit
                if token_count + len(tokens) > max_token_limit:
                    # If so, create a new DataFrame
                    df_list.append(current_df)
                    current_df = pd.DataFrame(columns=df.columns)
                    token_count = 0

                # Add the row to the current DataFrame
                new_df = pd.DataFrame([row], columns=df.columns)
                current_df = pd.concat([current_df, new_df]).reset_index(drop=True)
                # current_df = current_df.append(row, ignore_index=True)
                token_count += len(tokens)
            # Add the last DataFrame
            df_list.append(current_df)

            if len(df_list) > 0:
                chunks.extend(df_list)

        return chunks

    def _parse_entity(self, chunk, value):
        """
        Parses the entity to ground the responses and return values only present in the document chunk.
        """
        entity_info = {"value": None, "bbox": [0, 0, 0, 0]}
        words = []
        x_vals = []
        y_vals = []
        pattern = r" \d{2}\|\d{2}"
        result = regex_split(pattern, value)
        for i in range(0, len(result) // 2):
            entity_value = result[i * 2].strip()
            entity_segment = result[(i * 2) + 1].strip()

            if str(entity_segment) not in chunk["segment_id"].to_list():
                continue

            segment_value = chunk[chunk["segment_id"] == entity_segment][
                "segment"
            ].values[0]

            if entity_value not in segment_value:
                continue
            x_vals.append(
                chunk[chunk["segment_id"] == entity_segment]["x-mid"].values[0]
            )
            y_vals.append(
                chunk[chunk["segment_id"] == entity_segment]["y-mid"].values[0]
            )
            words.append(entity_value)
        if len(words) > 0:
            entity_info["value"] = " ".join(words)
            entity_info["bbox"] = [min(x_vals), min(y_vals), max(x_vals), max(y_vals)]
        return entity_info

    def _decode_response(self, json_response, chunk):
        """
        Calls the parse enity method on each response.
        """
        entities = {}
        for key in json_response:
            value = json_response[key]
            entities[key] = self._parse_entity(chunk, value)
        return entities

    def majority_voting(self, all_responses):
        """
        Finds the most frequent response for each entity.
        """
        merged = {}
        for response in all_responses:
            for k, v in response.items():
                if k not in merged:
                    merged[k] = v
                    continue

                if v["value"] is None:
                    continue

                if merged[k]["value"] is None:
                    merged[k] = v
                    continue

                merged[k]["value"] = Counter(
                    [v["value"], merged[k]["value"]]
                ).most_common(1)[0][0]
        return merged

    def postprocess(self, chunk, chunk_responses):
        grounded_responses = []
        for response in chunk_responses:
            grounded_responses.append(self._decode_response(response, chunk))
        final_response = self.majority_voting(grounded_responses)
        return final_response

    def postprocess_all_chunks(self, llm_responses):
        extracted_entities = []
        for chunk, chunk_responses in zip(self.chunks, llm_responses):
            extracted_entities.append(self.postprocess(chunk, chunk_responses))
        return extracted_entities
