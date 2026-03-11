# Release Scripts

All scripts must be run from the **marie-ai root directory**.

## Prerequisites

```bash
pip install twine wheel
npm install -g git-release-notes
```

Environment variables:
- `MARIE_SLACK_WEBHOOK` — required for Slack notifications on final/rc releases

## `update-version.sh` — bump version locally

Edits `marie/__init__.py` in-place. Does **not** commit, tag, publish, or push.

```bash
# Dev pre-release: 4.0.0 -> 4.0.0.dev<N> (N = commits since last tag)
./scripts/update-version.sh

# Release candidate: 4.0.0rc1 -> 4.0.0rc2
./scripts/update-version.sh rc

# Final: 4.0.0 -> 4.0.1
./scripts/update-version.sh final
```

## `release.sh` — full release pipeline

Reads the current version from `marie/__init__.py`, publishes to PyPI, and (for final/rc) generates a changelog, tags, commits, and sends a Slack notification.

Your local branch **must be in sync** with `origin` — the script will refuse to run otherwise.

```bash
# Dev pre-release: publish current version to PyPI, nothing else
./scripts/release.sh

# Release candidate: publish + changelog + tag + commit + Slack
./scripts/release.sh rc "<reason>" "<actor>"

# Final release: publish + changelog + tag + commit + Slack
./scripts/release.sh final "<reason>" "<actor>"
```

## Typical Workflows

### Final release

```bash
git checkout main && git pull
grep __version__ marie/__init__.py   # verify version, e.g. 4.0.0
./scripts/release.sh final "v4.0.0 GA release" "greg"
# Publishes 4.0.0 to PyPI, bumps __init__.py to 4.0.1, tags v4.0.0, commits
```

### Dev pre-release

```bash
./scripts/update-version.sh          # sets e.g. 4.0.0.dev37
./scripts/release.sh                 # publishes 4.0.0.dev37 to PyPI
git checkout -- marie/__init__.py    # revert the dev version bump
```

### Release candidate

```bash
./scripts/release.sh rc "testing rc1" "greg"
# Publishes current version to PyPI, bumps to next rc, tags, commits
```
