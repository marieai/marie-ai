#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MARIE_AI_ROOT="${MARIE_AI_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"

MARIE_WHEELS_DIR="${MARIE_WHEELS_DIR:-${MARIE_AI_ROOT}/wheels}"
MARIE_WHEELS_REPO="${MARIE_WHEELS_REPO:-marieai/marie-ai}"
MARIE_WHEELS_TAG="${MARIE_WHEELS_TAG:-wheels-torch212-cu130}"

WHEELS_README="${MARIE_WHEELS_DIR}/README.md"

usage() {
  cat <<'EOF'
Usage:
  tools/scripts/fetch-wheels.sh <command>

Provision the distributed wheels/ artifacts (etcd3, fastwer, Marie fairseq
fork, detectron2, FAISS CUDA 13, and the sdists) from a GitHub release instead
of tracking them in git or rebuilding them. Expected filenames and SHA256s
come from the generated inventory in wheels/README.md (maintained by
setup-py312-torch212-cu130.sh wheels-readme).

Commands:
  fetch     Download missing/mismatched artifacts and verify SHA256s. (default)
  verify    Check the artifacts present in wheels/ against the inventory.
  list      Print the expected artifacts and SHA256s.
  publish   Upload the local, hash-verified artifacts to the release (needs gh).

Environment overrides:
  MARIE_WHEELS_DIR=/path/to/marie-ai-checkout/wheels
  MARIE_WHEELS_REPO=marieai/marie-ai
  MARIE_WHEELS_TAG=wheels-torch212-cu130
  GITHUB_TOKEN=<token>   used by the curl fallback for private releases

Downloads prefer the gh CLI (works with private repos); curl is the fallback.
EOF
}

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

# Populate EXPECTED_FILES / EXPECTED_SHAS from the generated README inventory.
load_inventory() {
  [[ -f "${WHEELS_README}" ]] || die "missing ${WHEELS_README}"

  EXPECTED_FILES=()
  EXPECTED_SHAS=()

  local line file sha
  while IFS= read -r line; do
    file="$(awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' <<<"${line}")"
    sha="$(awk -F'|' '{gsub(/^[ \t]+|[ \t]+$/, "", $4); print $4}' <<<"${line}")"
    [[ "${sha}" =~ ^[0-9a-f]{64}$ ]] || continue
    [[ "${file}" == *.whl || "${file}" == *.tar.gz ]] || continue
    EXPECTED_FILES+=("${file}")
    EXPECTED_SHAS+=("${sha}")
  done < <(sed -n '/local-wheels-inventory:start/,/local-wheels-inventory:end/p' "${WHEELS_README}")

  [[ ${#EXPECTED_FILES[@]} -gt 0 ]] || die "no artifacts found in the ${WHEELS_README} inventory; run setup-py312-torch212-cu130.sh wheels-readme after a build"
}

sha_matches() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]]
}

download_asset() {
  local file="$1" dest="$2"
  if command -v gh >/dev/null 2>&1; then
    gh release download "${MARIE_WHEELS_TAG}" \
      --repo "${MARIE_WHEELS_REPO}" \
      --pattern "${file}" \
      --output "${dest}" \
      --clobber
  else
    local url="https://github.com/${MARIE_WHEELS_REPO}/releases/download/${MARIE_WHEELS_TAG}/${file}"
    local -a curl_args=(-fL --retry 3 -o "${dest}")
    [[ -n "${GITHUB_TOKEN:-}" ]] && curl_args+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    curl "${curl_args[@]}" "${url}"
  fi
}

cmd_list() {
  load_inventory
  local i
  for i in "${!EXPECTED_FILES[@]}"; do
    printf '%s  %s\n' "${EXPECTED_SHAS[$i]}" "${EXPECTED_FILES[$i]}"
  done
}

cmd_verify() {
  load_inventory
  local i file path failures=0
  for i in "${!EXPECTED_FILES[@]}"; do
    file="${EXPECTED_FILES[$i]}"
    path="${MARIE_WHEELS_DIR}/${file}"
    if [[ ! -f "${path}" ]]; then
      log "MISSING  ${file}"
      failures=$((failures + 1))
    elif sha_matches "${path}" "${EXPECTED_SHAS[$i]}"; then
      log "OK       ${file}"
    else
      log "MISMATCH ${file}"
      failures=$((failures + 1))
    fi
  done
  [[ ${failures} -eq 0 ]] || die "${failures} artifact(s) missing or mismatched"
  log "all wheels/ artifacts verified"
}

cmd_fetch() {
  load_inventory
  local i file sha path tmp fetched=0
  for i in "${!EXPECTED_FILES[@]}"; do
    file="${EXPECTED_FILES[$i]}"
    sha="${EXPECTED_SHAS[$i]}"
    path="${MARIE_WHEELS_DIR}/${file}"

    if [[ -f "${path}" ]] && sha_matches "${path}" "${sha}"; then
      log "OK (cached)  ${file}"
      continue
    fi

    log "fetching ${file} from ${MARIE_WHEELS_REPO}@${MARIE_WHEELS_TAG}"
    tmp="$(mktemp "${MARIE_WHEELS_DIR}/.${file}.XXXXXX")"
    if ! download_asset "${file}" "${tmp}"; then
      rm -f "${tmp}"
      die "download failed for ${file}"
    fi
    if ! sha_matches "${tmp}" "${sha}"; then
      rm -f "${tmp}"
      die "SHA256 mismatch for downloaded ${file}; release asset does not match the wheels/README.md inventory"
    fi
    mv "${tmp}" "${path}"
    log "verified     ${file}"
    fetched=$((fetched + 1))
  done
  log "done (${fetched} downloaded, $(( ${#EXPECTED_FILES[@]} - fetched )) already present)"
}

cmd_publish() {
  command -v gh >/dev/null 2>&1 || die "publish requires the gh CLI"
  load_inventory

  # Refuse to publish anything that does not match the inventory.
  cmd_verify

  if ! gh release view "${MARIE_WHEELS_TAG}" --repo "${MARIE_WHEELS_REPO}" >/dev/null 2>&1; then
    log "creating release ${MARIE_WHEELS_TAG} on ${MARIE_WHEELS_REPO}"
    gh release create "${MARIE_WHEELS_TAG}" \
      --repo "${MARIE_WHEELS_REPO}" \
      --title "Distributed wheels: torch 2.12 / cu130 / cp312" \
      --notes "Distributed wheels/ artifacts for the PyTorch 2.12.1+cu130 / Python 3.12 lane. SHA256 source of truth: wheels/README.md generated inventory. Provision with tools/scripts/fetch-wheels.sh."
  fi

  local i file
  for i in "${!EXPECTED_FILES[@]}"; do
    file="${EXPECTED_FILES[$i]}"
    log "uploading ${file}"
    gh release upload "${MARIE_WHEELS_TAG}" \
      --repo "${MARIE_WHEELS_REPO}" \
      --clobber \
      "${MARIE_WHEELS_DIR}/${file}"
  done
  log "published ${#EXPECTED_FILES[@]} artifact(s) to ${MARIE_WHEELS_REPO}@${MARIE_WHEELS_TAG}"
}

main() {
  local cmd="${1:-fetch}"
  case "${cmd}" in
    help|-h|--help) usage ;;
    fetch) cmd_fetch ;;
    verify) cmd_verify ;;
    list) cmd_list ;;
    publish) cmd_publish ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
