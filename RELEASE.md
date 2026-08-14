# Release & Version Identifier

Marie is shipped from two package management systems, PyPi and Docker Hub. This article clarifies the release cycle and version identifier behind each system.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [PyPi package versioning](#pypi-package-versioning)
- [Docker image versioning](#docker-image-versioning)
- [Manual Release Entrypoint](#manual-release-entrypoint)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## PyPi package versioning

We follow [PEP-440](https://www.python.org/dev/peps/pep-0440/), and a form of [semantic versioning](https://semver.org/) as explained above.

To install the latest final release into an application project:

```bash
uv add marie-ai
```

To install a particular final release:

```bash
uv add marie-ai==x.y.z
```

The term "final release" is relative to "developmental release" as described below.  

### Install Marie with Recommended Extensions

`uv add marie-ai` installs Marie into the current uv-managed project.

The recommended way of installing Marie is `uv add marie-ai`.

`"standard"` include extra dependencies that enables:
- Executor Hub + Docker support
- FastAPI + Websocket support (required when using `Flow(protocol='http')` or `Flow(protocol='websocket')`)
- the best compression via LZ4 algorithm
- the best async eventloop management via `uvloop`

The source checkout uses [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock) as the dependency source of truth.

##### Do I need "[standard]"?

Depends on how you use and distribute Marie. 

If you are using/distributing Marie as a microservice, use the Docker images or the uv extras for the target profile.

### Developmental releases versioning

One every master-merging event, we create early releases directly from source control which do not conflict with later project releases. The version identifier for development release is `x.y.z.devN`, which adds a suffix `.dev`, followed by a non-negative integer value `N`, which is reset on every release.

To install the latest development release:

```bash
uv add --prerelease allow marie-ai
```

### Version epochs


#### Release cycle and versioning
Marie is developed continuously by the community and core team. Updates are grouped and released at regular intervals to align with software development best practices.

Marie follows a form of numbered versioning. The version number of the product is a three-part value `x.y.z` where `x`, `y`, and `z` are the major, minor, and patch components respectively.

-   Patch release (`x.y.z` -> `x.y.(z+1)`): Contain bug fixes, new features and breaking changes. Released weekly on a Wednesday morning CET.
-   Minor release (`x.y.z -> x.(y+1).0`): Contain bug fixes, new features and breaking changes. Released monthly on the first Wednesday of the month CET. This release is more QA tested and considered more stable than a patch release.
-   Major release (`x.y.z -> (x+1).0.0`): Are released based on the development cycle of the Marie . There is no set scheduled for when these will occur.


The following example shows how Marie is released from 0.9 to 0.9.2 according to the schema we defined above.

|Event `e` | After `e`, `uv add marie-ai` | After `e`, `uv add --prerelease allow marie-ai` | After `e`, master `__init__.py` |
|--- |-----------------------------------| --- | --- |
| Release | 0.9.0                             | 0.9.0 | 0.9.1.dev0 |
| Master merging | 0.9.0                             | 0.9.1.dev0 | 0.9.1.dev1 |
| Master merging | 0.9.0                             | 0.9.1.dev1 | 0.9.1.dev2 |
| Master merging | 0.9.0                             | 0.9.1.dev2 | 0.9.1.dev3 |
| Release | 0.9.1                             | 0.9.1 | 0.9.2.dev0 |
| Master merging | 0.9.1                             | 0.9.2.dev0 | 0.9.2.dev1 |

## Docker image versioning

Our universal Docker image is ready-to-use on linux/amd64, linux/armv7+, linux/arm/v6, linux/arm64. The Docker image name always starts with `marie-ai/marie` followed by a tag composed of three parts:

```text
marie-ai/marie:{version}{python_version}{extra}
```

- `{version}`: The version of Marie. Possible values:
    - `latest`: the last release;
    - `master`: the master branch of `marie-ai/marie` repository;
    - `x.y.z`: the release of a particular version;
    - `x.y`: the alias to the last `x.y.z` patch release, i.e. `x.y` = `x.y.max(z)`;
- `{python_version}`: The Python version of the image. Possible values:
    - `-py312` for Python 3.12;
- `{extra}`: the extra dependency installed along with Marie. Possible values:
    - `-cpu`: Marie gateway profile built from the default `uv.lock` dependency set without torch;
    - `-cuda`: Marie CUDA profile built from the `cu130` extra in `uv.lock`;

Examples:

- `marie-ai/marie:0.9.6`: the `0.9.6` release with Python 3.7 and the entrypoint of `marie`.
- `marie-ai/marie:latest`: the latest release with Python 3.7 and the entrypoint of `marie`
- `marie-ai/marie:master`: the master with Python 3.7 and the entrypoint of `marie`

### Image alias and updates

| Event | Updated images | Aliases |
| --- | --- | --- |
| On Master Merge | `marie-ai/marie:master{python_version}{extra}` | |
| On `x.y.z` release | `marie-ai/marie:x.y.z{python_version}{extra}` | `marie-ai/marie:latest{python_version}{extra}`, `marie-ai/marie:x.y{python_version}{extra}` |

The PyTorch 2.12 release builds the Python 3.12 CPU gateway and CUDA images selected by `build.sh`.


## Manual Release Entrypoint

Use the release script from a checkout with an upstream branch:

```bash
./scripts/release.sh
```

The interactive menu selects the version bump, container profile, and whether
to publish. If the worktree is dirty, it also offers to stash tracked,
untracked, and staged work for the release and restore it afterward. The
release stops if the operator declines, the branch is behind its upstream,
version references are inconsistent, the image build fails, or the release tag
already exists.

The successful local sequence is:

1. Update every file managed by `scripts/update-version.sh`.
2. Create `chore(release): release X.Y.Z`.
3. Build and verify the selected images through `build.sh`.
4. Generate release notes from non-merge commit subjects since the previous
   release and store them in the annotated `vX.Y.Z` tag.

The tag is created after the image build so a failed build does not mark the
commit as released. If a build fails after the release commit is created, fix
the build and rerun with the exact current version.

Use non-interactive arguments for repeatable operation:

```bash
./scripts/release.sh patch --profile all
./scripts/release.sh --version 5.1.0 --profile marie-cuda
./scripts/release.sh rc --profile all --publish
./scripts/release.sh patch --profile all --dry-run
./scripts/release.sh patch --profile all --stash
```

`--stash` is explicit: `--yes` never enables it. The script records the exact
stash it creates and restores it with `git stash pop --index` on success,
failure, or cancellation. A clean restore drops that stash. If the release and
saved work modify the same lines, Git retains the stash and the script exits
with conflict-recovery instructions. Commit `scripts/release.sh` and
`scripts/update-version.sh` before using this mode so the running tool cannot
stash its own dependencies.

Stashed work is not included in the release notes because it is not committed.
Review the generated notes in the release plan or inspect a completed tag:

```bash
git tag -n99 v5.1.0
```

`--publish` pushes the release commit first, then the versioned container
images, and finally the Git tag. Without it, all artifacts remain local and
the script prints the corresponding push commands.

Use `--skip-build` only for package-only automation such as the Manual Release
GitHub workflow. Normal runtime releases should build both container profiles.
