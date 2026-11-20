# Release m3top

Use this runbook to publish the standalone Rust crate in `m3top/` from the
`marieai/marie-ai` repository. Source, tests, release tags, binary archives,
checksums, documentation, and the installer are owned by this repository.
Target machines install binaries without GitHub authentication or a Rust
toolchain.

m3top versions are independent of Marie-AI versions. For example, `m3top-v0.2.0`
can be released alongside Marie-AI `v5.1.0`. Never reuse a published m3top tag.
The publish workflow uses `--latest=false` so the component release does not
replace the latest Marie-AI core release on GitHub.

## Publishing permissions

`.github/workflows/release-m3top.yml` uses GitHub's built-in `GITHUB_TOKEN`.
The build jobs have read access; only the publish job grants `contents: write`.
The publish command requires an existing tag with `--verify-tag` and uses
`--latest=false`. No separate release token is required.

The public installer must also exist at `scripts/install-m3top.sh` on the Marie-AI `main` branch before advertising the one-line installation command.

## Prepare a release

Run these commands from the Marie-AI repository root and choose an unused
version. `0.2.1` is the next patch after the existing `0.2.0` release:

```bash
export M3TOP_VERSION=0.2.1
```

Update `version` in `m3top/Cargo.toml`. Refresh the lock file, then run the release checks:

```bash
cargo check --manifest-path m3top/Cargo.toml
cargo fmt --manifest-path m3top/Cargo.toml --check
cargo test --manifest-path m3top/Cargo.toml --locked
cargo clippy --manifest-path m3top/Cargo.toml --all-targets --locked -- -D warnings
cargo build --manifest-path m3top/Cargo.toml --release --locked
./m3top/target/release/m3top --version
```

Commit the intended m3top release scope and push that commit before tagging it.
The tagged commit must include `.github/workflows/release-m3top.yml` and the
matching crate and lock-file version. The Marie-AI core version remains
independent; use this runbook for m3top rather than `scripts/release.sh`.

## Publish

Create the annotated source tag in Marie-AI and push it to the Marie-AI remote:

```bash
git tag -a "m3top-v${M3TOP_VERSION}" -m "Release m3top ${M3TOP_VERSION}"
git push origin "m3top-v${M3TOP_VERSION}"
```

The tag triggers `.github/workflows/release-m3top.yml` in the same repository.
After both builds succeed, the workflow attaches a public GitHub Release and
its assets to that existing source tag.

The tag version must exactly match `m3top/Cargo.toml`. The workflow builds:

- `x86_64-unknown-linux-musl`
- `aarch64-unknown-linux-musl`

Each archive is accompanied by a SHA-256 checksum file.

## Monitor and verify

List the release runs:

```bash
gh run list --repo marieai/marie-ai --workflow "Release m3top"
```

Watch the run using the ID printed by the preceding command:

```bash
gh run watch <run-id> --repo marieai/marie-ai --exit-status
```

Confirm that the public Release and its assets exist:

```bash
gh release view "m3top-v${M3TOP_VERSION}" --repo marieai/marie-ai
```

Confirm that both target archives and both checksum files are present. Check
that GitHub's latest release still points to the Marie-AI core release:

```bash
gh release view --repo marieai/marie-ai --json tagName --jq .tagName
```

Test the published version through the public installer:

```bash
M3TOP_INSTALL_DIR="$(mktemp -d /tmp/m3top-release-check.XXXXXX)"
export M3TOP_INSTALL_DIR
curl -fsSL https://raw.githubusercontent.com/marieai/marie-ai/main/scripts/install-m3top.sh \
  | M3TOP_VERSION="${M3TOP_VERSION}" sh
"${M3TOP_INSTALL_DIR}/m3top" --version
```

Run the installer again without `M3TOP_VERSION` to verify automatic discovery
of the new component release. The installer searches `m3top-v*` releases rather
than using GitHub's latest core release.

## Recover a failed run

If the workflow failed before creating the Release because of a temporary
runner or permission problem, resolve that problem and retry the failed jobs
from the same tag:

```bash
gh run rerun <run-id> --repo marieai/marie-ai --failed
gh run watch <run-id> --repo marieai/marie-ai --exit-status
```

If a code or workflow change is needed, commit the correction and choose a new
unused patch version. Reruns use the original tagged source. Leave existing
tags in place.

If the Marie-AI Release already exists, the workflow intentionally refuses to replace it. Published versions are immutable. Correct the problem, increment the patch version, and publish a new tag instead of overwriting an existing Release.

## Historical releases

`m3top-v0.2.0` and earlier were built from the Studio repository. Their Studio
source tags, Marie-AI distribution tags, and public release assets remain
unchanged. After the source move, new m3top tags belong in Marie-AI and identify
the commit containing this crate.
