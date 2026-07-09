import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path

REPLACEMENTS = {
    "Requires-Dist: fvcore (<0.1.6,>=0.1.5)": "Requires-Dist: fvcore (>=0.1.5)",
    "Requires-Dist: fvcore<0.1.6,>=0.1.5": "Requires-Dist: fvcore>=0.1.5",
    "Requires-Dist: iopath (<0.1.10,>=0.1.7)": "Requires-Dist: iopath (>=0.1.7)",
    "Requires-Dist: iopath<0.1.10,>=0.1.7": "Requires-Dist: iopath>=0.1.7",
}
PATCHED_REQUIREMENTS = {
    "fvcore": {
        "Requires-Dist: fvcore (>=0.1.5)",
        "Requires-Dist: fvcore>=0.1.5",
    },
    "iopath": {
        "Requires-Dist: iopath (>=0.1.7)",
        "Requires-Dist: iopath>=0.1.7",
    },
}


def metadata_path() -> Path:
    try:
        dist = metadata.distribution("detectron2")
    except metadata.PackageNotFoundError:
        print("Error: detectron2 is not installed.", file=sys.stderr)
        sys.exit(1)

    for file in dist.files or []:
        if file.name == "METADATA" and file.parent.name.endswith(".dist-info"):
            return Path(dist.locate_file(file))

    print("Error: could not locate detectron2 METADATA.", file=sys.stderr)
    sys.exit(1)


def smoke_test() -> None:
    import fvcore
    import iopath
    import torch
    from detectron2 import _C
    from detectron2.layers import ROIAlign

    print("detectron2_ext", _C.__name__)
    print("fvcore", getattr(fvcore, "__version__", "unknown"))
    print("iopath", getattr(iopath, "__version__", "unknown"))

    if torch.cuda.is_available():
        x = torch.randn(1, 1, 8, 8, device="cuda")
        boxes = torch.tensor([[0, 0, 0, 7, 7]], dtype=torch.float32, device="cuda")
        out = ROIAlign((2, 2), 1.0, 2, True)(x, boxes)
        assert out.is_cuda, out.device
        print("detectron2_roi_align", tuple(out.shape), out.device)


def apply_patch(no_confirm: bool) -> None:
    path = metadata_path()
    content = path.read_text()

    old_requirements = [old for old in REPLACEMENTS if old in content]
    already_patched = all(
        any(requirement in content for requirement in requirements)
        for requirements in PATCHED_REQUIREMENTS.values()
    )
    if not old_requirements and already_patched:
        print(f"detectron2 metadata already patched: {path}")
        smoke_test()
        return

    if not old_requirements:
        print(
            "Error: expected detectron2 metadata requirements not found.",
            file=sys.stderr,
        )
        for requirement in REPLACEMENTS:
            print(requirement, file=sys.stderr)
        print(path, file=sys.stderr)
        sys.exit(1)

    smoke_test()

    if not no_confirm:
        response = (
            input(f"Patch detectron2 metadata at {path}? (yes/no): ").strip().lower()
        )
        if response not in {"yes", "y"}:
            print("Patch not applied.")
            return

    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)

    path.write_text(content)
    print(f"Patch applied successfully to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relax detectron2 fvcore/iopath metadata after runtime smoke."
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Apply patch without confirmation.",
        default=False,
    )
    args = parser.parse_args()
    apply_patch(args.no_confirm)
