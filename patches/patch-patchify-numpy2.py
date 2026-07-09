import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path

OLD_REQUIREMENTS = {
    "Requires-Dist: numpy (<2,>=1)",
    "Requires-Dist: numpy (>=1,<2)",
}
NEW_REQUIREMENT = "Requires-Dist: numpy (>=1,<3)"


def metadata_path() -> Path:
    try:
        dist = metadata.distribution("patchify")
    except metadata.PackageNotFoundError:
        print("Error: patchify is not installed.", file=sys.stderr)
        sys.exit(1)

    for file in dist.files or []:
        if file.name == "METADATA" and file.parent.name.endswith(".dist-info"):
            return Path(dist.locate_file(file))

    print("Error: could not locate patchify METADATA.", file=sys.stderr)
    sys.exit(1)


def smoke_test() -> None:
    import numpy as np
    from patchify import patchify, unpatchify

    image = np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    patches = patchify(image, (128, 128, 3), step=128)
    reconstructed = unpatchify(patches, image.shape)
    assert patches.shape == (4, 4, 1, 128, 128, 3), patches.shape
    assert np.array_equal(image, reconstructed)
    print("patchify numpy smoke", np.__version__, patches.shape)


def apply_patch(no_confirm: bool) -> None:
    path = metadata_path()
    content = path.read_text()

    if NEW_REQUIREMENT in content:
        print(f"patchify metadata already patched: {path}")
        smoke_test()
        return

    if not any(requirement in content for requirement in OLD_REQUIREMENTS):
        print("Error: expected patchify numpy requirement not found.", file=sys.stderr)
        print(path, file=sys.stderr)
        sys.exit(1)

    smoke_test()

    if not no_confirm:
        response = (
            input(f"Patch patchify metadata at {path}? (yes/no): ").strip().lower()
        )
        if response not in {"yes", "y"}:
            print("Patch not applied.")
            return

    for requirement in OLD_REQUIREMENTS:
        content = content.replace(requirement, NEW_REQUIREMENT)

    path.write_text(content)
    print(f"Patch applied successfully to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relax patchify 0.2.3 metadata after NumPy 2 runtime smoke."
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Apply patch without confirmation.",
        default=False,
    )
    args = parser.parse_args()
    apply_patch(args.no_confirm)
