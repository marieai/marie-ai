from repackage_tool import move_and_repackage_module, update_or_clone_repository

update_or_clone_repository(
    "https://github.com/run-llama/llama_index", destination="repo_clone", branch="main"
)

move_and_repackage_module(
    source_dir="repo_clone/llama-index-core",
    source_namespace="llama_index.core",
    target_namespace="marie.core",
    blacklist=[
        "tests",
        "examples",
        "command_line",
        "workflow",
        "llama_pack",
        "response_synthesizers",
        "download",
        "node_parser",
        "llama_dataset",
        "playground",
        "evaluation",
    ],
    output_dir="marie_output",
)
