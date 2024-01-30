import os
from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from numpy.linalg import norm

from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.jina.jina_embeddings import JinaEmbeddings
from marie.embeddings.sentence_transformers.sentence_transformers_embeddings import (
    SentenceTransformerEmbeddings,
)
from marie.ocr.util import meta_to_text
from marie.utils.utils import ensure_exists

cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))


def compare(embeddings_provider: EmbeddingsBase, lhs_file: str, rhs_file: str):
    """
    Compares the similarity of the two given files.

    Args:
        embeddings_provider (EmbeddingsBase): The embedding model to use.
        lhs_file (str): The path to the left-hand file.
        rhs_file (str): The path to the right-hand file.

    Returns:
        None
    """
    try:
        lhf_text = meta_to_text(lhs_file)
        rhf_text = meta_to_text(rhs_file)

        embedding_obj = embeddings_provider.get_embeddings(
            [
                lhf_text,
                rhf_text,
            ],
            truncation=True,
            max_length=4096,
        )
        assert embedding_obj is not None
        embeddings = embedding_obj.embeddings
        print(embeddings)
        print(cos_sim(embeddings[0], embeddings[1]))
    except Exception as e:
        raise e


def process_dir(embeddings_provider: EmbeddingsBase, src_dir: str, output_dir: str):
    """
    Processes all files in the given source directory and writes the extracted text to separate text files in the output directory.

    Args:
        embeddings_provider (EmbeddingsBase): The embedding model to use.
        src_dir (str): The path to the source directory.
        output_dir (str): The path to the output directory.

    Returns:
        None
    """
    root_asset_dir = os.path.expanduser(src_dir)
    output_path = os.path.expanduser(output_dir)

    for file_path in Path(root_asset_dir).rglob("*"):
        if not file_path.is_file():
            print(file_path)
            continue
        try:
            resolved_output_path = os.path.join(
                output_path, file_path.relative_to(root_asset_dir)
            )
            output_dir = os.path.dirname(resolved_output_path)
            filename = os.path.basename(resolved_output_path)
            name = os.path.splitext(filename)[0]
            os.makedirs(output_dir, exist_ok=True)

            print("file_path", file_path)
            print("resolved_output_path", resolved_output_path)
            text = meta_to_text(file_path, resolved_output_path)

            loader = TextLoader(resolved_output_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
            lc_docs = text_splitter.split_documents(documents)

            print(len(lc_docs))

            for doc in lc_docs:
                print(doc.metadata)
                print(doc.page_content)

                embedding_obj = embeddings_provider.get_embeddings(
                    [
                        doc.page_content,
                        doc.page_content,
                    ],
                    truncation=True,
                    max_length=4096,
                )
                assert embedding_obj is not None
                embeddings = embedding_obj.embeddings
                print(embeddings)
                print(cos_sim(embeddings[0], embeddings[1]))
            break

        except Exception as e:
            raise e


if __name__ == "__main__":
    ensure_exists("/tmp/marie/embeddings")

    embeddings_provider = JinaEmbeddings(
        model_name_or_path="hf://jinaai/jina-embeddings-v2-base-en"
    )

    embeddings_provider = SentenceTransformerEmbeddings()

    compare(
        embeddings_provider,
        "09052023_29373_1_797_3.json",
        "08312023_734839_1_218_2.json",
    )

    if False:
        process_dir(
            embeddings_provider,
            "~/datasets/private/corr-routing/ready_extended/jpmc_01-22-2024/ready/annotations/LEVEL_1/attorney",
            "/tmp/marie/embeddings",
        )
