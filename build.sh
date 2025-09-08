#!/bin/bash

# Marie AI Docker Build Script
# This script builds the Marie AI Docker image with profile support and standard pip configuration
set -e
CPU_COUNT=$(grep -c ^processor /proc/cpuinfo)
CPU_COUNT=$((CPU_COUNT-1))

readonly DEFAULT_VERSION="4.0.9"
readonly VERSION="${MARIE_VERSION:-$DEFAULT_VERSION}"

declare -A PROFILES=(
    ["marie-gateway"]="marieai/marie-gateway:${VERSION}-cpu:./Dockerfiles/cpu-312.slim.Dockerfile"
    ["marie-cuda"]="marieai/marie:${VERSION}-cuda:./Dockerfiles/cuda-312.Dockerfile"
)

# Default configuration (will be overridden by profile selection)
DOCKERFILE_PATH=""
IMAGE_NAME=""
IMAGE_TAG=""
readonly PIP_TAG="standard"
FULL_IMAGE_NAME=""

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
    exec 2>&2 # flush
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
    exec 2>&2 # flush
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exec 2>&2 # flush
}

log_prompt() {
    echo -e "\033[0;34m[PROMPT]\033[0m $1" >&2
    exec 2>&2 # flush
}

show_profiles() {
    echo >&2
    log_info "Marie AI Docker Builder (Version: ${VERSION})"
    log_info "Available build profiles:"
    echo "1) marie-gateway      - MarieAI Gateway (CPU) -> marieai/marie-gateway:4.0.0-cpu" >&2
    echo "2) marie-cuda         - MarieAI Core (CUDA) -> marieai/marie:4.0.0-cuda" >&2
    echo "3) all                - Build all profiles" >&2
    echo "4) exit               - Exit without building" >&2
    echo >&2
}

parse_profile_config() {
    local profile_key=$1
    local config="${PROFILES[$profile_key]}"

    IFS=':' read -r IMAGE_NAME IMAGE_TAG DOCKERFILE_PATH <<< "$config"
    FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version=*)
                VERSION="${1#*=}"
                shift
                ;;
            -v|--version)
                if [[ -n "${2:-}" && ! "${2:-}" =~ ^- ]]; then
                    VERSION="$2"
                    shift 2
                else
                    log_error "Version argument requires a value"
                    exit 1
                fi
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                # This is the profile selection, handle it in select_profile
                break
                ;;
        esac
    done

    PROFILES["marie-gateway-cpu"]="marieai/marie-gateway:${VERSION}-cpu:./Dockerfiles/cpu-312.slim.Dockerfile"
    PROFILES["marie-cuda"]="marieai/marie:${VERSION}-cuda:./Dockerfiles/cuda-312.Dockerfile"
}

show_help() {
    echo "Marie AI Docker Build Script"
    echo
    echo "Usage: $0 [OPTIONS] [PROFILE]"
    echo
    echo "Options:"
    echo "  -v, --version VERSION    Set the version tag (default: $DEFAULT_VERSION)"
    echo "  -h, --help              Show this help message"
    echo
    echo "Profiles:"
    echo "  marie-gateway-cpu       Build MarieAI Gateway CPU image"
    echo "  marie-cuda              Build MarieAI Core CUDA image"
    echo "  all                     Build all profiles"
    echo "  1, 2, 3                 Numeric profile selection"
    echo
    echo "Examples:"
    echo "  $0                                    # Interactive mode with default version"
    echo "  $0 --version 4.1.0 marie-cuda        # Build CUDA image with version 4.1.0"
    echo "  $0 -v 4.1.0 all                      # Build all images with version 4.1.0"
    echo "  MARIE_VERSION=4.1.0 $0 marie-cuda    # Using environment variable"
}

# Select build profile
select_profile() {
    local profile_choice=""
    local remaining_args=("$@")

    # Check if profile provided as command line argument
    if [[ ${#remaining_args[@]} -gt 0 ]]; then
        case "${remaining_args[0]}" in
            "marie-gateway-cpu"|"1")
                profile_choice="marie-gateway-cpu"
                ;;
            "marie-cuda"|"2")
                profile_choice="marie-cuda"
                ;;
            "all"|"3")
                profile_choice="all"
                ;;
            *)
                log_error "Invalid profile: ${remaining_args[0]}"
                show_profiles
                exit 1
                ;;
        esac
    else
        # Interactive selection
        while [[ -z "$profile_choice" ]]; do
            show_profiles
            log_prompt "Please select a build profile (1-4): "
            read -r choice

            case "$choice" in
                1|"marie-gateway-cpu")
                    profile_choice="marie-gateway-cpu"
                    ;;
                2|"marie-cuda")
                    profile_choice="marie-cuda"
                    ;;
                3|"all")
                    profile_choice="all"
                    ;;
                4|"exit")
                    log_info "Exiting without building."
                    exit 0
                    ;;
                *)
                    log_error "Invalid choice. Please select 1-4."
                    ;;
            esac
        done
    fi

    echo "$profile_choice"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Execute post-commit hook
execute_post_commit_hook() {
    local hook_path="./hooks/post-commit"

    if [[ -f "$hook_path" ]]; then
        log_info "Executing post-commit hook..."
        "$hook_path"
    else
        log_warn "Post-commit hook not found at: $hook_path"
    fi
}

# Build Docker image
build_image() {
    local dockerfile_path=$1
    local full_image_name=$2

    if [[ ! -f "$dockerfile_path" ]]; then
        log_error "Dockerfile not found at: $dockerfile_path"
        return 1
    fi

    log_info "Building Docker image: $full_image_name"
    log_info "Using Dockerfile: $dockerfile_path"
    log_info "PIP tag: $PIP_TAG"
    log_info "Version: $VERSION"

    # Debug: List what's actually in the build context
    log_info "Contents of build context:"
    ls -la patches/ wheels/ || log_warn "Some directories might be missing"

    DOCKER_BUILDKIT=0 docker build . \
        --network=host \
        --cpuset-cpus="0-$CPU_COUNT" \
        --build-arg PIP_TAG="$PIP_TAG" \
        --build-arg VCS_REF=$(git rev-parse HEAD) \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg MARIE_VERSION="$VERSION" \
        --build-arg TARGETPLATFORM=linux/amd64 \
        --no-cache \
        -f "$dockerfile_path" \
        -t "$full_image_name"

    if [[ $? -eq 0 ]]; then
        log_info "Docker image built successfully: $full_image_name"
        return 0
    else
        log_error "Docker image build failed: $full_image_name"
        return 1
    fi
}

# Verify image creation
verify_image() {
    local full_image_name=$1

    log_info "Verifying image creation: $full_image_name"

    local image_id
    image_id=$(docker images --format='{{.ID}}' "$full_image_name" | head -1)

    if [[ -n "$image_id" ]]; then
        log_info "Image verified successfully with ID: $image_id"
        return 0
    else
        log_error "Image verification failed - no image found: $full_image_name"
        return 1
    fi
}

# Build single profile
build_single_profile() {
    local profile_key=$1

    log_info "Building profile: $profile_key (Version: $VERSION)"
    parse_profile_config "$profile_key"

    if build_image "$DOCKERFILE_PATH" "$FULL_IMAGE_NAME"; then
        verify_image "$FULL_IMAGE_NAME"
        return $?
    else
        return 1
    fi
}

# Build all profiles
build_all_profiles() {
    local failed_builds=()
    local successful_builds=()

    log_info "Building all profiles (Version: $VERSION)..."

    for profile_key in "${!PROFILES[@]}"; do
        log_info "Starting build for profile: $profile_key"

        if build_single_profile "$profile_key"; then
            successful_builds+=("$profile_key")
        else
            failed_builds+=("$profile_key")
        fi

        echo "----------------------------------------"
    done

    # Summary
    log_info "Build Summary (Version: $VERSION):"
    if [[ ${#successful_builds[@]} -gt 0 ]]; then
        log_info "Successful builds:"
        for profile in "${successful_builds[@]}"; do
            parse_profile_config "$profile"
            echo "  ✓ $profile -> $FULL_IMAGE_NAME"
        done
    fi

    if [[ ${#failed_builds[@]} -gt 0 ]]; then
        log_error "Failed builds:"
        for profile in "${failed_builds[@]}"; do
            echo "  ✗ $profile"
        done
        return 1
    fi

    return 0
}

# Main execution
main() {
    log_info "Starting Marie AI Docker build process..."

    # Parse arguments first to get version
    parse_arguments "$@"

    # Remove parsed arguments, keep profile selection
    local remaining_args=()
    local skip_next=false

    for arg in "$@"; do
        if [[ "$skip_next" == "true" ]]; then
            skip_next=false
            continue
        fi

        case "$arg" in
            --version=*|--help|-h)
                # Already handled
                ;;
            -v|--version)
                skip_next=true
                ;;
            *)
                remaining_args+=("$arg")
                ;;
        esac
    done

    check_prerequisites
#    execute_post_commit_hook

    local selected_profile
    selected_profile=$(select_profile "${remaining_args[@]}")

    if [[ "$selected_profile" == "all" ]]; then
        build_all_profiles
    else
        build_single_profile "$selected_profile"
    fi

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        log_info "Build process completed successfully"
    else
        log_error "Build process failed"
    fi

    exit $exit_code
}

# Execute main function
main "$@"