#!/usr/bin/env python3
"""
Verify that hash_frames_fast produces consistent hashes for identical images.

Usage:
    source ~/environments/marie-3.12/bin/activate
    python verify_hash_frames.py
    python verify_hash_frames.py /path/to/custom/directory
"""

import hashlib
import sys
from pathlib import Path

# Import marie modules - api.docs first to avoid circular import
from marie.api.docs import MarieDoc  # noqa: F401 - needed to avoid circular import
from marie.utils.docs import load_image
from marie.utils.image_utils import hash_frames_fast


def analyze_directory(base_dir):
    """Analyze a generator directory for duplicate detection"""
    base_dir = Path(base_dir)

    print(f"\n{'='*70}")
    print(f"Analyzing directory: {base_dir.name}")
    print(f"{'='*70}")

    tif_files = sorted(base_dir.glob("*.tif"))
    if not tif_files:
        print("No TIF files found in directory")
        return

    print(f"\nFound {len(tif_files)} TIF file(s):\n")

    for tif_file in tif_files:
        print(f"  {tif_file}")

    print()
    results = []
    for tif_file in tif_files:
        # Compute file MD5
        with open(tif_file, 'rb') as f:
            file_md5 = hashlib.md5(f.read()).hexdigest()

        # Compute hash_frames_fast using existing method
        loaded, frames = load_image(str(tif_file))
        if not loaded:
            print(f"  ERROR: Could not load {tif_file.name}")
            continue

        frame_hash = hash_frames_fast(frames)

        results.append(
            {
                'filename': tif_file.name,
                'file_md5': file_md5,
                'frame_hash': frame_hash,
                'num_pages': len(frames),
                'shapes': [f.shape for f in frames],
            }
        )

        print(f"File: {tif_file.name}")
        print(f"  File MD5:         {file_md5}")
        print(f"  hash_frames_fast: {frame_hash}")
        print(f"  Pages: {len(frames)}, Shapes: {[f.shape for f in frames]}")
        print(f"  Matches dir name: {frame_hash == base_dir.name}")
        print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Directory name: {base_dir.name}")

    # Check for duplicates by file content
    file_md5s = [r['file_md5'] for r in results]
    unique_file_md5s = set(file_md5s)
    print(f"\nFile content analysis:")
    print(f"  Total TIF files: {len(file_md5s)}")
    print(f"  Unique file contents: {len(unique_file_md5s)}")

    if len(unique_file_md5s) < len(file_md5s):
        print(f"  DUPLICATE FILE CONTENT DETECTED!")
        for md5 in unique_file_md5s:
            dupes = [r['filename'] for r in results if r['file_md5'] == md5]
            if len(dupes) > 1:
                print(f"     Files with same content (MD5: {md5}):")
                for d in dupes:
                    print(f"       - {d}")

    # Check hash consistency
    frame_hashes = [r['frame_hash'] for r in results]
    unique_frame_hashes = set(frame_hashes)
    print(f"\nhash_frames_fast analysis:")
    print(f"  Unique frame hashes: {len(unique_frame_hashes)}")

    if len(unique_frame_hashes) == 1:
        the_hash = frame_hashes[0]
        if the_hash == base_dir.name:
            print(f"  All files produce same hash: {the_hash}")
            print(f"  Hash matches directory name")
            print(f"\n  CONCLUSION: hash_frames_fast is working correctly.")
            print(
                f"              These are duplicate SOURCE FILES with identical content."
            )
        else:
            print(f"  Hash mismatch with directory name!")
            print(f"     Files hash: {the_hash}")
            print(f"     Dir name:   {base_dir.name}")
    else:
        print(f"  Multiple different hashes found:")
        for h in unique_frame_hashes:
            files = [r['filename'] for r in results if r['frame_hash'] == h]
            print(f"     {h}: {files}")


def scan_all_generators(base_dir):
    """Scan all generator directories and find duplicates across the entire directory"""
    base_dir = Path(base_dir)

    print(f"\n{'='*70}")
    print(f"SCANNING ALL GENERATORS IN: {base_dir}")
    print(f"{'='*70}")

    # Find all subdirectories that look like hash directories (32 char hex)
    all_dirs = [d for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 32]
    print(f"\nFound {len(all_dirs)} generator directories")

    # Collect all files and their hashes
    all_files = []
    duplicates_within_dir = []
    hash_mismatches = []

    for i, gen_dir in enumerate(sorted(all_dirs)):
        tif_files = list(gen_dir.glob("*.tif"))
        if not tif_files:
            continue

        if (i + 1) % 50 == 0:
            print(f"  Processing directory {i+1}/{len(all_dirs)}...")

        dir_results = []
        for tif_file in tif_files:
            try:
                # Compute file MD5
                with open(tif_file, 'rb') as f:
                    file_md5 = hashlib.md5(f.read()).hexdigest()

                # Compute hash_frames_fast
                loaded, frames = load_image(str(tif_file))
                if not loaded:
                    continue

                frame_hash = hash_frames_fast(frames)

                entry = {
                    'dir_name': gen_dir.name,
                    'filename': tif_file.name,
                    'full_path': str(tif_file),
                    'file_md5': file_md5,
                    'frame_hash': frame_hash,
                    'num_pages': len(frames),
                }
                all_files.append(entry)
                dir_results.append(entry)

                # Check if hash matches directory name
                if frame_hash != gen_dir.name:
                    hash_mismatches.append(entry)

            except Exception as e:
                print(f"  ERROR processing {tif_file}: {e}")

        # Check for duplicates within this directory
        if len(dir_results) > 1:
            md5s = [r['file_md5'] for r in dir_results]
            if len(set(md5s)) < len(md5s):
                duplicates_within_dir.append(
                    {
                        'dir': gen_dir.name,
                        'files': [r['filename'] for r in dir_results],
                        'md5': dir_results[0]['file_md5'],
                    }
                )

    # Summary
    print(f"\n{'='*70}")
    print("GLOBAL ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal files analyzed: {len(all_files)}")
    print(f"Total directories: {len(all_dirs)}")

    # Find duplicate file contents across ALL directories
    print(f"\n--- Duplicate File Contents (same MD5 in different directories) ---")
    md5_to_files = {}
    for entry in all_files:
        md5 = entry['file_md5']
        if md5 not in md5_to_files:
            md5_to_files[md5] = []
        md5_to_files[md5].append(entry)

    cross_dir_dupes = {
        md5: files
        for md5, files in md5_to_files.items()
        if len(set(f['dir_name'] for f in files)) > 1
    }

    if cross_dir_dupes:
        print(
            f"Found {len(cross_dir_dupes)} file contents appearing in multiple directories:"
        )
        for md5, files in list(cross_dir_dupes.items())[:10]:  # Show first 10
            dirs = set(f['dir_name'] for f in files)
            print(f"\n  MD5: {md5}")
            print(f"  Appears in {len(dirs)} directories:")
            for f in files[:5]:  # Show first 5 files
                print(f"    - {f['full_path']}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")
    else:
        print("No duplicate file contents found across different directories")

    # Duplicates within same directory
    print(f"\n--- Duplicate Files Within Same Directory ---")
    if duplicates_within_dir:
        print(f"Found {len(duplicates_within_dir)} directories with duplicate files:")
        for dup in duplicates_within_dir[:10]:
            print(f"\n  Directory: {dup['dir']}")
            print(f"  MD5: {dup['md5']}")
            print(f"  Files: {dup['files']}")
    else:
        print("No duplicate files within same directory")

    # Hash mismatches (potential hash_frames_fast issue)
    print(f"\n--- Hash Mismatches (frame_hash != directory name) ---")
    if hash_mismatches:
        print(f"FOUND {len(hash_mismatches)} FILES WITH HASH MISMATCH:")
        for entry in hash_mismatches[:20]:
            print(f"\n  File: {entry['full_path']}")
            print(f"  Dir name:    {entry['dir_name']}")
            print(f"  Frame hash:  {entry['frame_hash']}")
            print(f"  File MD5:    {entry['file_md5']}")
    else:
        print("All files have matching hashes - hash_frames_fast is working correctly!")

    # Check for hash collisions (different content, same hash)
    print(f"\n--- Hash Collisions (different content, same frame_hash) ---")
    hash_to_md5s = {}
    for entry in all_files:
        h = entry['frame_hash']
        if h not in hash_to_md5s:
            hash_to_md5s[h] = set()
        hash_to_md5s[h].add(entry['file_md5'])

    collisions = {h: md5s for h, md5s in hash_to_md5s.items() if len(md5s) > 1}
    if collisions:
        print(f"FOUND {len(collisions)} HASH COLLISIONS:")
        for h, md5s in list(collisions.items())[:10]:
            print(f"\n  Frame hash: {h}")
            print(f"  Different file MD5s: {md5s}")
            for entry in all_files:
                if entry['frame_hash'] == h:
                    print(f"    - {entry['full_path']} (MD5: {entry['file_md5']})")
    else:
        print(
            "No hash collisions found - hash_frames_fast produces unique hashes for different content!"
        )


def main():
    default_dir = "/home/gbugaj/.marie/generators_WITH_DUPLICATED_GENDIR/db81e247d8a9f691673d4a1f9bf947a9"

    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = default_dir

    if not Path(target_dir).exists():
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)

    # Check if this is a generators directory or a single hash directory
    if Path(target_dir).name == "generators" or "generators" in Path(target_dir).name:
        scan_all_generators(target_dir)
    elif len(Path(target_dir).name) == 32:
        # Single hash directory
        analyze_directory(target_dir)
    else:
        # Assume it's a parent directory, look for generators subdirectory
        generators_dir = Path(target_dir) / "generators"
        if generators_dir.exists():
            scan_all_generators(generators_dir)
        else:
            # Try scanning directly
            scan_all_generators(target_dir)


if __name__ == "__main__":
    main()
