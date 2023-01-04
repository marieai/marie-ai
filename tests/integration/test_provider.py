import os

from marie.utils.utils import FileSystem, ensure_exists

if __name__ == '__main__':
    img_path = 'word/0001.png'
    base_dir = FileSystem.get_share_directory()
    path = os.path.abspath(os.path.join(base_dir, img_path))

    print(base_dir)
    print(path)
