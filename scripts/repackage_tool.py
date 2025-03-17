import argparse
import os
import re
import shutil
import subprocess


def update_or_clone_repository(repo_url, destination="repo_clone", branch="main"):
    """
    Updates the existing Git repository if found, otherwise clones it.

    Args:
        repo_url (str): The GitHub repository URL.
        destination (str): The directory where the repo will be cloned.
        branch (str): The branch to checkout.
    """
    if os.path.exists(destination):
        print(f"Updating existing repository in {destination}...")
        try:
            subprocess.run(["git", "-C", destination, "pull"], check=True)
            subprocess.run(["git", "-C", destination, "checkout", branch], check=True)
        except subprocess.CalledProcessError:
            print("Failed to update. Removing and cloning again...")
            shutil.rmtree(destination)
            subprocess.run(
                ["git", "clone", "--branch", branch, repo_url, destination], check=True
            )
    else:
        print(f"Cloning repository: {repo_url} into {destination}")
        subprocess.run(
            ["git", "clone", "--branch", branch, repo_url, destination], check=True
        )


def transform_imports(file_path, source_namespace, target_namespace):
    """
    Transforms import statements from `source_namespace` to `target_namespace`.

    Args:
        file_path (str): The file to transform.
        source_namespace (str): Original namespace (e.g., `llama_index.core`).
        target_namespace (str): Target namespace (e.g., `marie.core`).
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Replace imports dynamically
    content = re.sub(
        fr"\bfrom\s+{re.escape(source_namespace)}(\.[\w\.]*)?\s+import\s+",
        fr"from {target_namespace}\1 import ",
        content,
    )

    content = re.sub(
        fr"\bimport\s+{re.escape(source_namespace)}(\.[\w\.]*)?",
        fr"import {target_namespace}\1",
        content,
    )

    # Write back transformed content
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def move_and_repackage_module(
    source_dir, source_namespace, target_namespace, blacklist=None, output_dir="output"
):
    """
    Moves subdirectories from source namespace to the target namespace and transforms imports.

    Args:
        source_dir (str): Directory containing the source module.
        source_namespace (str): Original namespace (e.g., `llama_index.core`).
        target_namespace (str): Target namespace (e.g., `marie.core`).
        blacklist (list): List of modules or subdirectories to exclude.
        output_dir (str): Directory where the repackaged module will be stored.
    """
    if blacklist is None:
        blacklist = []

    source_core_dir = os.path.join(source_dir, *source_namespace.split("."))
    if not os.path.exists(source_core_dir):
        raise FileNotFoundError(f"Source directory `{source_core_dir}` not found.")

    target_core_dir = os.path.join(output_dir, *target_namespace.split("."))
    os.makedirs(target_core_dir, exist_ok=True)

    print(f"Repackaging `{source_namespace}` into `{target_namespace}` namespace...")

    for root, dirs, files in os.walk(source_core_dir):
        rel_path = os.path.relpath(root, source_core_dir)

        # Skip blacklisted directories
        if any(blacklisted in rel_path for blacklisted in blacklist):
            print(f"Skipping blacklisted module: {rel_path}")
            continue

        target_path = os.path.join(target_core_dir, rel_path)
        os.makedirs(target_path, exist_ok=True)

        for file in files:
            if file.endswith(".py"):  # Only process Python files
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_path, file)
                shutil.copy2(src_file, dst_file)
                transform_imports(dst_file, source_namespace, target_namespace)

    print(f"Repackaging completed. Output located at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Repackage a repository module into a new namespace."
    )
    parser.add_argument("--repo-url", type=str, required=True, help="Repository URL")
    parser.add_argument("--branch", type=str, default="main", help="Branch to clone")
    parser.add_argument(
        "--source-namespace",
        type=str,
        required=True,
        help="Source namespace (e.g., llama_index.core)",
    )
    parser.add_argument(
        "--target-namespace",
        type=str,
        required=True,
        help="Target namespace (e.g., marie.core)",
    )
    parser.add_argument(
        "--blacklist", type=str, nargs="*", help="Modules to exclude", default=[]
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="repo_clone",
        help="Temporary directory for cloning",
    )

    args = parser.parse_args()

    # Clone or update the repository
    update_or_clone_repository(
        args.repo_url, destination=args.temp_dir, branch=args.branch
    )

    # Move and repackage the module
    move_and_repackage_module(
        source_dir=args.temp_dir,
        source_namespace=args.source_namespace,
        target_namespace=args.target_namespace,
        blacklist=args.blacklist,
        output_dir=args.output_dir,
    )

    # Clean up temporary directory
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)


if __name__ == "__main__":
    main()
