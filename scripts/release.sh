#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
UPDATE_VERSION="${SCRIPT_DIR}/update-version.sh"
BUILD_SCRIPT="${REPO_ROOT}/build.sh"

VERSION_SPEC=""
PROFILE=""
SKIP_BUILD=false
PUBLISH=false
PUBLISH_SET=false
ASSUME_YES=false
DRY_RUN=false
TARGET_VERSION=""
STASH_CHANGES=false
STASH_OID=""
STASH_REF=""
WORKTREE_WAS_DIRTY=false
NOTES_FILE=""
RELEASE_BASE=""
RELEASE_BASE_LABEL=""
RELEASE_COMMIT_COUNT=0

usage() {
    cat <<'EOF'
Usage: scripts/release.sh [OPTIONS] [VERSION]

Create a Marie release commit, build versioned container images, and tag it.
Run without arguments for the interactive release menu.

VERSION may be an exact version or one of: major, minor, patch, final, rc.

Options:
  -v, --version VERSION    Version or bump type
  -p, --profile PROFILE    all, marie-gateway-cpu, or marie-cuda
      --skip-build         Create the release commit and tag without containers
      --publish            Push artifacts, create the GitHub Release and PR
      --no-publish         Keep the commit, images, and tag local
      --stash              Stash tracked and untracked work, then restore it
  -n, --dry-run            Show the release plan without changing anything
  -y, --yes                Skip the final confirmation
  -h, --help               Show this help

Examples:
  ./scripts/release.sh
  ./scripts/release.sh patch --profile all
  ./scripts/release.sh --version 5.1.0 --profile marie-cuda
  ./scripts/release.sh rc --profile all --publish
  ./scripts/release.sh patch --profile all --stash
EOF
}

log() {
    printf '[release] %s\n' "$*"
}

die() {
    printf '[release] ERROR: %s\n' "$*" >&2
    exit 1
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -v|--version)
                [[ -n "${2:-}" ]] || die "$1 requires a value"
                VERSION_SPEC=$2
                shift 2
                ;;
            --version=*)
                VERSION_SPEC=${1#*=}
                shift
                ;;
            -p|--profile)
                [[ -n "${2:-}" ]] || die "$1 requires a value"
                PROFILE=$2
                shift 2
                ;;
            --profile=*)
                PROFILE=${1#*=}
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --publish)
                PUBLISH=true
                PUBLISH_SET=true
                shift
                ;;
            --no-publish)
                PUBLISH=false
                PUBLISH_SET=true
                shift
                ;;
            --stash)
                STASH_CHANGES=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -y|--yes)
                ASSUME_YES=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                die "unknown option: $1"
                ;;
            *)
                [[ -z "$VERSION_SPEC" ]] || die "only one version may be specified"
                VERSION_SPEC=$1
                shift
                ;;
        esac
    done

    [[ $# -eq 0 ]] || die "unexpected argument: $1"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

github_cli_install_hint() {
    case "$(uname -s)" in
        Darwin)
            printf '[release] Install it with: brew install gh\n' >&2
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                printf '[release] Install it with: sudo apt install gh\n' >&2
            elif command -v dnf >/dev/null 2>&1; then
                printf '[release] Install it with: sudo dnf install gh\n' >&2
            elif command -v pacman >/dev/null 2>&1; then
                printf '[release] Install it with: sudo pacman -S github-cli\n' >&2
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            printf '[release] Install it with: winget install --id GitHub.cli\n' >&2
            ;;
    esac
    printf '[release] Installation guide: https://cli.github.com/manual/installation\n' >&2
}

require_github_cli() {
    if ! command -v gh >/dev/null 2>&1; then
        printf '[release] ERROR: GitHub CLI (gh) is required for --publish\n' >&2
        github_cli_install_hint
        exit 1
    fi
    if ! gh auth status --hostname github.com >/dev/null 2>&1; then
        die "GitHub CLI is not authenticated; run: gh auth login --hostname github.com"
    fi
    if ! gh repo view --json nameWithOwner >/dev/null 2>&1; then
        die "GitHub CLI cannot access the repository configured for this checkout"
    fi
}

require_clean_worktree() {
    local status
    status=$(git status --porcelain=v1 --untracked-files=normal)
    if [[ -n "$status" ]]; then
        printf '%s\n' "$status" >&2
        die "commit, stash, or remove worktree changes before releasing"
    fi
}

require_version_files_clean_for_stash() {
    local status
    local -a release_files

    mapfile -t release_files < <("$UPDATE_VERSION" --files)
    status=$(git status --porcelain=v1 --untracked-files=normal -- "${release_files[@]}")
    if [[ -n "$status" ]]; then
        printf '%s\n' "$status" >&2
        die "version-managed files cannot be stashed during a release; commit or restore them first"
    fi
}

restore_worktree() {
    local exit_status=$1
    local current_oid
    local restore_status

    trap - EXIT

    if [[ -n "$NOTES_FILE" ]]; then
        rm -f -- "$NOTES_FILE"
    fi

    if [[ -n "$STASH_OID" ]]; then
        current_oid=$(git rev-parse -q --verify "${STASH_REF}^{commit}" 2>/dev/null || true)
        if [[ "$current_oid" != "$STASH_OID" ]]; then
            printf '[release] ERROR: release command ended, but the saved worktree stash moved\n' >&2
            printf '[release] Saved stash commit: %s\n' "$STASH_OID" >&2
            printf '[release] Restore it with: git stash apply --index %s\n' "$STASH_OID" >&2
            exit_status=1
        else
            log "Restoring stashed worktree changes"
            set +e
            git stash pop --index "$STASH_REF"
            restore_status=$?
            set -e
            if (( restore_status != 0 )); then
                printf '[release] ERROR: release command ended, but stashed changes could not be restored cleanly\n' >&2
                printf '[release] The stash was retained as %s (%s)\n' "$STASH_REF" "$STASH_OID" >&2
                printf '[release] Resolve the conflicts and verify the restored work, then drop it with: git stash drop %s\n' "$STASH_REF" >&2
                exit_status=1
            else
                log "Restored the pre-release worktree and staged state"
            fi
        fi
    fi

    exit "$exit_status"
}

prepare_worktree() {
    local status
    local answer
    local new_stash
    local previous_stash
    local stash_status

    status=$(git status --porcelain=v1 --untracked-files=normal)
    if [[ -z "$status" ]]; then
        return
    fi

    WORKTREE_WAS_DIRTY=true
    printf '%s\n' "$status" >&2

    if [[ "$STASH_CHANGES" == false ]]; then
        if [[ "$ASSUME_YES" == false && -t 0 ]]; then
            read -r -p 'Stash these changes for the release and restore them afterward? [y/N] ' answer
            if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
                STASH_CHANGES=true
            fi
        fi
    fi

    if [[ "$STASH_CHANGES" == false ]]; then
        die "worktree is dirty; commit the changes or rerun with --stash"
    fi

    require_version_files_clean_for_stash

    if [[ "$DRY_RUN" == true ]]; then
        log "Dry run: worktree changes would be stashed and restored"
        return
    fi

    if ! git diff --quiet HEAD -- scripts/release.sh scripts/update-version.sh; then
        die "commit the release tooling before using --stash; the running script cannot stash its own dependencies"
    fi

    previous_stash=$(git rev-parse -q --verify refs/stash 2>/dev/null || true)
    set +e
    git stash push --include-untracked -m "marie release ${VERSION_SPEC:-interactive}"
    stash_status=$?
    set -e

    new_stash=$(git rev-parse -q --verify 'stash@{0}^{commit}' 2>/dev/null || true)
    if [[ -n "$new_stash" && "$new_stash" != "$previous_stash" ]]; then
        STASH_REF='stash@{0}'
        STASH_OID=$new_stash
    fi

    if (( stash_status != 0 )); then
        if [[ -n "$STASH_OID" ]]; then
            die "Git saved the worktree but could not clean it for release; restoring the saved changes"
        fi
        die "Git could not stash and clean the worktree"
    fi

    [[ -n "$STASH_OID" ]] || die "could not identify the worktree stash"
    require_clean_worktree
    log "Saved worktree changes as ${STASH_REF} (${STASH_OID:0:12})"
}

check_source_state() {
    local branch=$1
    local upstream=$2
    local counts
    local local_ahead
    local remote_ahead

    if [[ "$DRY_RUN" == false ]]; then
        log "Refreshing ${upstream}"
        git fetch --quiet || die "could not refresh ${upstream}"
    fi

    counts=$(git rev-list --left-right --count "HEAD...${upstream}")
    read -r local_ahead remote_ahead <<< "$counts"
    if (( remote_ahead > 0 )); then
        die "${branch} is behind ${upstream} by ${remote_ahead} commit(s)"
    fi
    if (( local_ahead > 0 )); then
        log "Warning: ${branch} is ahead of ${upstream} by ${local_ahead} commit(s)"
    fi
}

select_release_base() {
    local current=$1
    local target=$2
    local target_major=${target%%.*}
    local tag
    local base_index=0
    local -a version_commits

    while IFS= read -r tag; do
        if [[ "$tag" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)((a|b|rc)[0-9]+)?$ ]] && \
                [[ "${BASH_REMATCH[1]}" == "$target_major" ]]; then
            RELEASE_BASE=$tag
            RELEASE_BASE_LABEL=$tag
            return
        fi
    done < <(git tag --merged HEAD --sort=-version:refname)

    mapfile -t version_commits < <(git log --format='%H' -- marie/_version.py)
    if [[ "$target" == "$current" ]]; then
        base_index=1
    fi

    if (( ${#version_commits[@]} > base_index )); then
        RELEASE_BASE=${version_commits[$base_index]}
        RELEASE_BASE_LABEL="version anchor $(git rev-parse --short "$RELEASE_BASE")"
    else
        RELEASE_BASE=""
        RELEASE_BASE_LABEL="repository start"
    fi
}

generate_release_notes() {
    local current=$1
    local target=$2
    local revision=HEAD
    local -a entries

    select_release_base "$current" "$target"
    if [[ -n "$RELEASE_BASE" ]]; then
        revision="${RELEASE_BASE}..HEAD"
    fi

    mapfile -t entries < <(
        git log --no-merges \
            --invert-grep --grep='^chore(release): release ' \
            --format='- %h %s (%an)' "$revision"
    )
    RELEASE_COMMIT_COUNT=${#entries[@]}
    NOTES_FILE=$(mktemp "${TMPDIR:-/tmp}/marie-release-notes.XXXXXX")

    {
        printf 'Marie AI %s\n\n' "$target"
        printf 'Changes since %s:\n' "$RELEASE_BASE_LABEL"
        if (( RELEASE_COMMIT_COUNT == 0 )); then
            printf '%s\n' '- No committed changes.'
        else
            printf '%s\n' "${entries[@]}"
        fi
    } > "$NOTES_FILE"
}

resolve_version() {
    "$UPDATE_VERSION" --resolve "$1"
}

choose_version() {
    local current=$1
    local choice
    local exact

    printf '\nMarie release version (current: %s)\n' "$current" >&2
    printf '1) patch  -> %s\n' "$(resolve_version patch)" >&2
    printf '2) minor  -> %s\n' "$(resolve_version minor)" >&2
    printf '3) major  -> %s\n' "$(resolve_version major)" >&2
    printf '4) rc     -> %s\n' "$(resolve_version rc)" >&2
    printf '5) exact version\n' >&2
    printf '6) exit\n' >&2
    read -r -p 'Select a release version (1-6): ' choice

    case "$choice" in
        1) printf 'patch\n' ;;
        2) printf 'minor\n' ;;
        3) printf 'major\n' ;;
        4) printf 'rc\n' ;;
        5)
            read -r -p 'Version: ' exact
            printf '%s\n' "$exact"
            ;;
        6) exit 0 ;;
        *) die "invalid version selection: $choice" ;;
    esac
}

choose_profile() {
    local choice

    printf '\nContainer profiles\n' >&2
    printf '1) all                Gateway CPU and executor CUDA\n' >&2
    printf '2) marie-gateway-cpu  Gateway CPU only\n' >&2
    printf '3) marie-cuda         Executor CUDA only\n' >&2
    read -r -p 'Select a build profile (1-3): ' choice

    case "$choice" in
        1) printf 'all\n' ;;
        2) printf 'marie-gateway-cpu\n' ;;
        3) printf 'marie-cuda\n' ;;
        *) die "invalid profile selection: $choice" ;;
    esac
}

validate_profile() {
    case "$1" in
        all|marie-gateway-cpu|marie-cuda) ;;
        *) die "invalid build profile: $1" ;;
    esac
}

image_names() {
    local version=$1
    local profile=$2

    case "$profile" in
        all)
            printf 'marieai/marie-gateway:%s-cpu\n' "$version"
            printf 'marieai/marie:%s-cuda\n' "$version"
            ;;
        marie-gateway-cpu)
            printf 'marieai/marie-gateway:%s-cpu\n' "$version"
            ;;
        marie-cuda)
            printf 'marieai/marie:%s-cuda\n' "$version"
            ;;
    esac
}

show_plan() {
    local current=$1
    local target=$2
    local tag=$3
    local branch=$4

    printf '\nRelease plan\n'
    printf '  Version:  %s -> %s\n' "$current" "$target"
    if [[ "$current" == "$target" ]]; then
        printf '  Commit:   reuse current HEAD\n'
    else
        printf '  Commit:   chore(release): release %s\n' "$target"
    fi
    printf '  Tag:      %s (after successful build)\n' "$tag"
    printf '  Notes:    %s commit(s) since %s\n' \
        "$RELEASE_COMMIT_COUNT" "$RELEASE_BASE_LABEL"
    if [[ "$WORKTREE_WAS_DIRTY" == true ]]; then
        printf '  Worktree: stash and restore tracked, staged, and untracked changes\n'
    fi
    if [[ "$SKIP_BUILD" == true ]]; then
        printf '  Images:   skipped\n'
    else
        while IFS= read -r image; do
            printf '  Image:    %s\n' "$image"
        done < <(image_names "$target" "$PROFILE")
    fi
    if [[ "$PUBLISH" == true ]]; then
        printf '  Publish:  commit, images, tag, then GitHub Release\n'
        printf '  GitHub:   Marie AI %s from annotated %s\n' "$target" "$tag"
        if [[ "$branch" != "main" ]]; then
            printf '  PR:       %s -> main after publication; merge commit required\n' "$branch"
        fi
    else
        printf '  Publish:  no; artifacts remain local\n'
    fi
    printf '\nRelease notes preview\n'
    head -n 23 "$NOTES_FILE"
    if (( RELEASE_COMMIT_COUNT > 20 )); then
        printf -- '- ... %s more commit(s)\n' "$((RELEASE_COMMIT_COUNT - 20))"
    fi
    printf '\n'
}

ensure_release_pr() {
    local branch=$1
    local tag=$2
    local pr_url

    if [[ "$branch" == "main" ]]; then
        return
    fi

    pr_url=$(gh pr list \
        --head "$branch" \
        --base main \
        --state open \
        --json url \
        --jq '.[0].url')
    if [[ -n "$pr_url" ]]; then
        log "Release PR: ${pr_url}"
        return
    fi

    log "Creating pull request from ${branch} to main"
    gh pr create \
        --base main \
        --head "$branch" \
        --title "Merge ${tag} into main" \
        --body "Published ${tag} from ${branch}. Merge with Create a merge commit so the tagged release commit remains in main history; do not squash or rebase."
}

confirm_release() {
    local answer

    if [[ "$ASSUME_YES" == true ]]; then
        return
    fi
    read -r -p 'Continue with this release? [y/N] ' answer
    [[ "$answer" == "y" || "$answer" == "Y" ]] || exit 0
}

commit_version() {
    local current=$1
    local target=$2
    local -a release_files

    if [[ "$current" == "$target" ]]; then
        "$UPDATE_VERSION" --check
        log "Release files already use ${target}; reusing current HEAD"
        return
    fi

    "$UPDATE_VERSION" "$target"
    "$UPDATE_VERSION" --check
    mapfile -t release_files < <("$UPDATE_VERSION" --files)
    git add -- "${release_files[@]}"

    git diff --cached --check
    git diff --cached --quiet && die "version update produced no staged changes"
    git diff --quiet || die "version update left unexpected unstaged changes"
    [[ -z "$(git ls-files --others --exclude-standard)" ]] || \
        die "version update left unexpected untracked files"

    git commit -m "chore(release): release ${target}"
}

publish_release() {
    local branch=$1
    local tag=$2
    local remote
    local merge_ref
    local remote_branch
    local -a gh_args

    remote=$(git config --get "branch.${branch}.remote")
    merge_ref=$(git config --get "branch.${branch}.merge")
    [[ -n "$remote" && -n "$merge_ref" ]] || \
        die "${branch} does not have a configured push target"
    remote_branch=${merge_ref#refs/heads/}

    log "Pushing release commit to ${remote}/${remote_branch}"
    git push "$remote" "HEAD:${remote_branch}"

    if [[ "$SKIP_BUILD" == false ]]; then
        while IFS= read -r image; do
            log "Pushing ${image}"
            docker push "$image"
        done < <(image_names "$TARGET_VERSION" "$PROFILE")
    fi

    log "Publishing release tag ${tag}"
    git push "$remote" "refs/tags/${tag}"

    gh_args=(
        release create "$tag"
        --verify-tag
        --title "Marie AI ${TARGET_VERSION}"
        --notes-from-tag
    )
    if [[ "$TARGET_VERSION" =~ (a|b|rc)[0-9]+$ ]]; then
        gh_args+=(--prerelease)
    fi
    log "Creating GitHub Release for ${tag}"
    gh "${gh_args[@]}"
    ensure_release_pr "$branch" "$tag"
}

main() {
    local current_version
    local branch
    local upstream
    local tag

    parse_arguments "$@"
    cd "$REPO_ROOT"

    require_command git
    require_command python3
    [[ -x "$UPDATE_VERSION" ]] || die "not executable: $UPDATE_VERSION"
    [[ -x "$BUILD_SCRIPT" ]] || die "not executable: $BUILD_SCRIPT"
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "not in a Git worktree"
    git var GIT_AUTHOR_IDENT >/dev/null 2>&1 || \
        die "configure user.name and user.email before releasing"
    branch=$(git branch --show-current)
    [[ -n "$branch" ]] || die "releases cannot be created from detached HEAD"
    if [[ "$PUBLISH" == true ]]; then
        require_github_cli
    fi
    trap 'restore_worktree $?' EXIT
    prepare_worktree

    current_version=$("$UPDATE_VERSION" --current)
    if [[ -z "$VERSION_SPEC" ]]; then
        [[ -t 0 ]] || die "--version is required when input is not interactive"
        VERSION_SPEC=$(choose_version "$current_version")
    fi
    TARGET_VERSION=$(resolve_version "$VERSION_SPEC")
    readonly TARGET_VERSION
    [[ "$TARGET_VERSION" != *.dev* ]] || die "development versions are not release tags"

    if [[ "$SKIP_BUILD" == false ]]; then
        if [[ -z "$PROFILE" ]]; then
            [[ -t 0 ]] || die "--profile is required when input is not interactive"
            PROFILE=$(choose_profile)
        fi
        validate_profile "$PROFILE"
        if [[ "$DRY_RUN" == false ]]; then
            require_command docker
            if [[ "$PROFILE" == "all" || "$PROFILE" == "marie-cuda" ]]; then
                require_command uv
            fi
        fi
    fi

    if [[ "$PUBLISH_SET" == false && "$ASSUME_YES" == false && -t 0 ]]; then
        local publish_answer
        read -r -p 'Publish the commit, images, tag, and GitHub Release after building? [y/N] ' publish_answer
        if [[ "$publish_answer" == "y" || "$publish_answer" == "Y" ]]; then
            PUBLISH=true
            require_github_cli
        fi
    fi

    upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null) || \
        die "${branch} does not have an upstream"
    check_source_state "$branch" "$upstream"

    tag="v${TARGET_VERSION}"
    if git show-ref --verify --quiet "refs/tags/${tag}"; then
        die "tag already exists: ${tag}"
    fi

    "$UPDATE_VERSION" --check
    generate_release_notes "$current_version" "$TARGET_VERSION"
    show_plan "$current_version" "$TARGET_VERSION" "$tag" "$branch"
    if [[ "$DRY_RUN" == true ]]; then
        log "Dry run complete; no files, images, commits, or tags were changed"
        return
    fi
    confirm_release

    commit_version "$current_version" "$TARGET_VERSION"

    if [[ "$SKIP_BUILD" == false ]]; then
        "$BUILD_SCRIPT" --version "$TARGET_VERSION" "$PROFILE"
        require_clean_worktree
        "$UPDATE_VERSION" --check
    fi

    git tag -a "$tag" -F "$NOTES_FILE"
    log "Created ${tag} at $(git rev-parse --short HEAD)"

    if [[ "$PUBLISH" == true ]]; then
        publish_release "$branch" "$tag"
        log "Published ${tag}"
    else
        log "Release is local. Publish when ready with:"
        log "  git push"
        if [[ "$SKIP_BUILD" == false ]]; then
            while IFS= read -r image; do
                log "  docker push ${image}"
            done < <(image_names "$TARGET_VERSION" "$PROFILE")
        fi
        log "  git push $(git config --get "branch.${branch}.remote") ${tag}"
    fi
}

main "$@"
