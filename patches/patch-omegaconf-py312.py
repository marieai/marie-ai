import os
import sys


def apply_patch():
    # Determine the path to the omegaconf.py file
    try:
        import omegaconf

        target_file = os.path.join(os.path.dirname(omegaconf.__file__), "omegaconf.py")
    except ImportError:
        print("Error: omegaconf module is not installed.")
        sys.exit(1)

    # Check if the file exists
    if not os.path.isfile(target_file):
        print(f"Error: {target_file} does not exist.")
        sys.exit(1)

    # Check if the patch is already present
    with open(target_file, "r") as file:
        content = file.read()
        if (
            "from dataclasses import _MISSING_TYPE" in content
            and "if isinstance(obj, _MISSING_TYPE):" in content
        ):
            print("Patch is already present in the file.")
            sys.exit(0)

    # Prompt the user to verify if they want to apply the patch
    response = (
        input(f"Do you want to apply the patch to {target_file}? (yes/no): ")
        .strip()
        .lower()
    )
    if response not in ["yes", "y"]:
        print("Patch not applied.")
        sys.exit(0)

    # Define the patch content
    patch_content = """
            from dataclasses import _MISSING_TYPE
            if isinstance(obj, _MISSING_TYPE):
                return OmegaConf.create({}, parent=parent, flags=flags)
    """

    # Apply the patch
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "if obj is _DEFAULT_MARKER_:" in line:
            lines.insert(i + 2, patch_content)
            break

    # Write the patched content back to the file
    with open(target_file, "w") as file:
        file.write("\n".join(lines))

    print(f"Patch applied successfully to {target_file}")


if __name__ == "__main__":
    apply_patch()
