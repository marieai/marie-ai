import base64
import difflib
import io
import json
import os

import streamlit as st
import streamlit_shortcuts
from PIL import Image
from rich import print as rprint
from rich.console import Console
from rich.theme import Theme

src_dir = os.path.expanduser("~/tmp/ocr-diffs/v3/ocr2")
output_dir = os.path.expanduser("~/tmp/ocr-diffs/reviewed-v3")

os.makedirs(output_dir, exist_ok=True)

json_files = [
    os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".json")
]
json_files.sort()

json_files = [
    os.path.join(src_dir, f)
    for f in os.listdir(src_dir)
    if f.endswith(".json")
    and not os.path.exists(os.path.join(output_dir, os.path.splitext(f)[0] + ".txt"))
]
json_files.sort()


def load_json(file_index):
    # Load the JSON file
    with open(json_files[file_index]) as f:
        data = json.load(f)
    return data


def display_image(base64_string, scale=1):
    decoded_image = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(decoded_image))
    image = image.resize((int(image.width * scale), int(image.height * scale)))
    st.image(image)


def save_image_and_text(image_data, text, base_name):
    # Save the image and text to output_dir
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    image.save(os.path.join(output_dir, f"{base_name}.png"))
    with open(os.path.join(output_dir, f"{base_name}.txt"), "w") as f:
        f.write(text)


def main():
    if "current_file_index" not in st.session_state:
        st.session_state.current_file_index = 0
    if "selected_word" not in st.session_state:
        st.session_state.selected_word = ""

    # Load the JSON file
    if st.session_state.current_file_index < len(json_files):
        data = load_json(st.session_state.current_file_index)
    else:
        st.write("No more files to process.")
        return

    # Define the actions for the shortcut keys
    def on_next():
        st.session_state.current_file_index = min(
            len(json_files) - 1, st.session_state.current_file_index + 1
        )

    def on_prev():
        st.session_state.current_file_index = max(
            0, st.session_state.current_file_index - 1
        )

    display_image(data["snippet"], 3)
    base_name = os.path.splitext(
        os.path.basename(json_files[st.session_state.current_file_index])
    )[0]

    col1, col2 = st.columns(2)
    with col1:
        # if st.button("Previous"):
        #     on_prev()
        streamlit_shortcuts.button(
            "Previous", on_click=on_prev, shortcut="Shift+ArrowUp"
        )

    with col2:
        # if st.button("Next"):
        #     on_next()
        streamlit_shortcuts.button("Next", on_click=on_next, shortcut="Shift+ArrowDown")
    # add a separator
    st.markdown("---")

    # Create three columns
    col1, col2, col3 = st.columns(3)

    def accept_word(snippet, word, base_name):
        def callback():
            # st.write(f"You accepted {word}")
            st.session_state.selected_word = word
            save_image_and_text(snippet, word, base_name)
            st.session_state.current_file_index += 1

        return callback

    with col1:
        word_1 = st.text_input("Word 1", data["word1"]["text"], key="Word1")
        streamlit_shortcuts.button(
            "Accept Word 1",
            on_click=accept_word(data["snippet"], word_1, base_name),
            shortcut="Shift+ArrowLeft",
        )
        # if st.button("X"):
        #     st.write("You accepted Word 1:", word_1)
        #     save_image_and_text(data["snippet"], word_1, base_name)
        #     st.session_state.current_file_index += 1

    with col2:
        word_2 = st.text_input("Word 2", data["word2"]["text"], key="Word2")

        streamlit_shortcuts.button(
            "Accept Word 2",
            on_click=accept_word(data["snippet"], word_2, base_name),
            shortcut="Shift+ArrowRight",
        )

        # if st.button("Accept Word 2"):
        #     st.write("You accepted Word 2:", word_2)
        #     save_image_and_text(data["snippet"], word_2, base_name)
        #     st.session_state.current_file_index += 1

    st.write("Last selected word:", st.session_state.selected_word)

    # Add a legend for the shortcut keys
    st.markdown(
        """
    ## Shortcut Keys
    - **Shift + ArrowUp**: Previous
    - **Shift + ArrowDown**: Next
    - **Shift + ArrowLeft**: Accept Word 1
    - **Shift + ArrowRight**: Accept Word 2
    """
    )

    with col3:
        st.text(word_1)
        st.text(word_2)

        # Create a diff line
        diff = difflib.ndiff(word_1, word_2)
        diff_line = "\n".join(diff)
        st.text("Diff:")
        st.text(diff_line)


if __name__ == "__main__":
    main()
