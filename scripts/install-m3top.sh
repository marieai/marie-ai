#!/bin/sh
set -eu

release_repository="marieai/marie-ai"
release_api="https://api.github.com/repos/${release_repository}/releases"
install_dir="${M3TOP_INSTALL_DIR:-${HOME}/.local/bin}"

die() {
  printf 'm3top install: %s\n' "$*" >&2
  exit 1
}

for command in curl tar sha256sum install; do
  command -v "${command}" >/dev/null 2>&1 || die "${command} is required"
done

case "$(uname -s)" in
  Linux) ;;
  *) die "only Linux releases are currently available" ;;
esac

case "$(uname -m)" in
  x86_64 | amd64) target="x86_64-unknown-linux-musl" ;;
  aarch64 | arm64) target="aarch64-unknown-linux-musl" ;;
  *) die "unsupported architecture: $(uname -m)" ;;
esac

version="${M3TOP_VERSION:-}"
if [ -z "${version}" ]; then
  version="$(
    curl -fsSL --retry 3 \
      -H 'Accept: application/vnd.github+json' \
      -H 'X-GitHub-Api-Version: 2022-11-28' \
      "${release_api}?per_page=100" \
      | sed -n 's/^[[:space:]]*"tag_name":[[:space:]]*"m3top-v\([^"]*\)".*/\1/p' \
      | sed -n '1p'
  )"
fi

version="${version#m3top-v}"
version="${version#v}"
case "${version}" in
  '' | *[!0-9A-Za-z.-]*) die "invalid release version: ${version}" ;;
esac

case "${install_dir}" in
  /*) ;;
  *) die "M3TOP_INSTALL_DIR must be an absolute path" ;;
esac

archive="m3top-${version}-${target}.tar.gz"
package="m3top-${version}-${target}"
download_url="https://github.com/${release_repository}/releases/download/m3top-v${version}"
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/m3top-install.XXXXXX")"
trap 'rm -rf "${temp_dir}"' EXIT HUP INT TERM

printf 'Downloading m3top %s for %s\n' "${version}" "${target}"
curl -fsSL --retry 3 -o "${temp_dir}/${archive}" "${download_url}/${archive}"
curl -fsSL --retry 3 -o "${temp_dir}/${archive}.sha256" "${download_url}/${archive}.sha256"

(
  cd "${temp_dir}"
  sha256sum -c "${archive}.sha256"
  tar -xzf "${archive}"
)

mkdir -p "${install_dir}"
install -m 0755 "${temp_dir}/${package}/m3top" "${install_dir}/m3top"

printf 'Installed %s\n' "${install_dir}/m3top"
"${install_dir}/m3top" --version

case ":${PATH}:" in
  *:"${install_dir}":*) ;;
  *) printf 'Add %s to PATH to run m3top directly.\n' "${install_dir}" ;;
esac
