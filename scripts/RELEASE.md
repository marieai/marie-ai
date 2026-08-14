# Release Scripts

Run release commands from the `marie-ai` repository root.

## Create a release

Use the interactive entry point:

```bash
./scripts/release.sh
```

It checks the worktree and upstream state, updates every managed version
reference, creates the release commit, builds the selected containers, and
creates an annotated tag after the build succeeds. The annotated tag contains
release notes generated from non-merge commit subjects. Publishing the commit,
images, tag, GitHub Release, and pull-request handoff is optional. Releases may
run from `develop` or a release development branch.

Use arguments for automation:

```bash
./scripts/release.sh patch --profile all
./scripts/release.sh --version 5.1.0 --profile marie-cuda
./scripts/release.sh rc --profile all --publish
./scripts/release.sh patch --profile all --dry-run
./scripts/release.sh patch --profile all --stash
```

Use `--stash` to explicitly save tracked, staged, and untracked work before the
release and restore it afterward. Interactive mode offers the same choice when
the worktree is dirty; `--yes` does not select it. If restoration conflicts,
the script retains the stash and reports the recovery steps. Commit the release
scripts themselves before using this mode. Version-managed files cannot be
stashed during a release.

`--publish` requires `gh` and an authenticated GitHub session. When `gh` is
missing, the script prints an installation command and the official installation
guide. Authenticate with `gh auth login --hostname github.com`. After publishing
from a branch other than `main`, the script reuses or creates a pull request to
`main`. Merge it with a merge commit, not squash or rebase, to preserve the
tagged commit in `main` history.

See [`../RELEASE.md`](../RELEASE.md) for the complete lifecycle and recovery
instructions.

## Inspect or update versions

`update-version.sh` remains the lower-level version primitive used by the
release script:

```bash
./scripts/update-version.sh --current
./scripts/update-version.sh --check
./scripts/update-version.sh --resolve patch
./scripts/update-version.sh patch
```

Use it directly only when intentionally updating version references without
creating a release commit, container image, or tag.
