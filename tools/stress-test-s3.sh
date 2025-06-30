#!/bin/bash

# S3 Stress Test Script - Robust version
# Usage: ./stress-test-s3.sh [options]

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Default configuration
DEFAULT_HOST="localhost:8000"
DEFAULT_ACCESS_KEY="MARIEACCESSKEY"
DEFAULT_SECRET_KEY="MARIESECRETACCESSKEY"
DEFAULT_BUCKET="warp-benchmark"
DEFAULT_CONCURRENT=20
DEFAULT_OBJECTS=1000
DEFAULT_SIZE="1MiB"
DEFAULT_DURATION="30s"
DEFAULT_OUTPUT_DIR="./benchmark-results"
DEFAULT_TLS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Print usage information
usage() {
    cat << EOF
S3 Stress Test Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --host HOST             S3 endpoint host (default: $DEFAULT_HOST)
    -a, --access-key KEY        Access key (default: $DEFAULT_ACCESS_KEY)
    -s, --secret-key KEY        Secret key (default: $DEFAULT_SECRET_KEY)
    -b, --bucket BUCKET         Bucket name (default: $DEFAULT_BUCKET)
    -c, --concurrent NUM        Concurrent operations (default: $DEFAULT_CONCURRENT)
    -o, --objects NUM           Number of objects (default: $DEFAULT_OBJECTS)
    -z, --size SIZE             Object size (default: $DEFAULT_SIZE)
    -d, --duration DURATION     Test duration (default: $DEFAULT_DURATION)
    -t, --tls                   Enable TLS (default: disabled)
    -O, --output-dir DIR        Output directory (default: $DEFAULT_OUTPUT_DIR)
    -T, --test-type TYPE        Test type: mixed|put|get|delete|stat (default: mixed)
    -r, --retain-bucket         Don't delete bucket after test
    -v, --verbose               Verbose output
    -C, --cleanup-only          Only cleanup previous test buckets
    --help                      Show this help message

EXAMPLES:
    # Basic test
    $0

    # High load test
    $0 --concurrent 50 --objects 5000 --size 10MiB

    # Quick connectivity test
    $0 --objects 10 --size 1KiB --test-type put

    # Production-like test with TLS
    $0 --host my-s3.example.com:443 --tls --concurrent 100

    # Cleanup only
    $0 --cleanup-only

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--host)
                HOST="$2"
                shift 2
                ;;
            -a|--access-key)
                ACCESS_KEY="$2"
                shift 2
                ;;
            -s|--secret-key)
                SECRET_KEY="$2"
                shift 2
                ;;
            -b|--bucket)
                BUCKET="$2"
                shift 2
                ;;
            -c|--concurrent)
                CONCURRENT="$2"
                shift 2
                ;;
            -o|--objects)
                OBJECTS="$2"
                shift 2
                ;;
            -z|--size)
                SIZE="$2"
                shift 2
                ;;
            -d|--duration)
                DURATION="$2"
                shift 2
                ;;
            -t|--tls)
                TLS=true
                shift
                ;;
            -O|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -T|--test-type)
                TEST_TYPE="$2"
                shift 2
                ;;
            -r|--retain-bucket)
                RETAIN_BUCKET=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -C|--cleanup-only)
                CLEANUP_ONLY=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."

    local missing_tools=()

    # Check for required tools
    for tool in curl wget tar; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install missing tools and try again"
        exit 1
    fi

    # Check disk space (need at least 100MB for warp binary and logs)
    local available_space
    available_space=$(df . | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 102400 ]]; then  # 100MB in KB
        log_warning "Low disk space available ($(($available_space/1024))MB). Tests may fail."
    fi

    log_success "System requirements check passed"
}

# Install or update warp tool
install_warp() {
    local warp_path="/usr/local/bin/warp"
    local temp_dir="/tmp/warp-install-$$"

    log_info "Checking warp installation..."

    # Check if warp exists and get version
    if command -v warp &> /dev/null; then
        local current_version
        current_version=$(warp --version 2>/dev/null | head -1 || echo "unknown")
        log_info "Found warp: $current_version"

        # If version is recent enough, skip installation
        if [[ "$current_version" =~ warp.version ]]; then
            log_success "Warp is already installed and ready"
            return 0
        fi
    fi

    log_info "Installing/updating warp..."

    # Create temporary directory
    mkdir -p "$temp_dir"
    cd "$temp_dir"

    # Download with retries
    local download_url="https://github.com/minio/warp/releases/latest/download/warp_Linux_x86_64.tar.gz"
    local max_retries=3
    local retry=0

    while [[ $retry -lt $max_retries ]]; do
        if wget -q --timeout=30 "$download_url"; then
            break
        fi
        retry=$((retry + 1))
        log_warning "Download attempt $retry failed, retrying..."
        sleep 2
    done

    if [[ $retry -eq $max_retries ]]; then
        log_error "Failed to download warp after $max_retries attempts"
        cleanup_temp "$temp_dir"
        exit 1
    fi

    # Extract and install
    if tar -xzf warp_Linux_x86_64.tar.gz && [[ -f warp ]]; then
        if sudo mv warp "$warp_path" && sudo chmod +x "$warp_path"; then
            log_success "Warp installed successfully"
        else
            log_error "Failed to install warp (permission denied?)"
            cleanup_temp "$temp_dir"
            exit 1
        fi
    else
        log_error "Failed to extract warp binary"
        cleanup_temp "$temp_dir"
        exit 1
    fi

    cleanup_temp "$temp_dir"
}

# Cleanup temporary directory
cleanup_temp() {
    local temp_dir="$1"
    cd /
    rm -rf "$temp_dir" 2>/dev/null || true
}

# Test S3 connectivity
test_connectivity() {
    log_info "Testing S3 connectivity..."

    local protocol="http"
    [[ "$TLS" == "true" ]] && protocol="https"

    local endpoint="$protocol://$HOST"

    # Test basic connectivity
    if ! curl -f -s --connect-timeout 10 --max-time 30 "$endpoint" >/dev/null; then
        log_error "Cannot connect to S3 endpoint: $endpoint"
        log_info "Please check:"
        log_info "  - S3 server is running"
        log_info "  - Host and port are correct"
        log_info "  - Network connectivity"
        exit 1
    fi

    log_success "S3 endpoint is reachable"
}

# Cleanup old test buckets
cleanup_buckets() {
    log_info "Cleaning up old test buckets..."

    local cleanup_output
    cleanup_output=$(warp delete \
        --host "$HOST" \
        --access-key "$ACCESS_KEY" \
        --secret-key "$SECRET_KEY" \
        $([ "$TLS" == "true" ] || echo "--tls=false") \
        --bucket "warp-benchmark*" \
        2>&1 || true)

    if [[ "$VERBOSE" == "true" ]]; then
        echo "$cleanup_output"
    fi

    log_success "Cleanup completed"
}

# Validate test parameters
validate_parameters() {
    log_info "Validating test parameters..."

    # Validate concurrent workers
    if [[ ! "$CONCURRENT" =~ ^[0-9]+$ ]] || [[ "$CONCURRENT" -lt 1 ]] || [[ "$CONCURRENT" -gt 1000 ]]; then
        log_error "Invalid concurrent value: $CONCURRENT (must be 1-1000)"
        exit 1
    fi

    # Validate object count
    if [[ ! "$OBJECTS" =~ ^[0-9]+$ ]] || [[ "$OBJECTS" -lt 1 ]]; then
        log_error "Invalid objects value: $OBJECTS (must be positive integer)"
        exit 1
    fi

    # Validate object size format
    if [[ ! "$SIZE" =~ ^[0-9]+[KMGT]?i?B?$ ]]; then
        log_error "Invalid size format: $SIZE (examples: 1KB, 1MiB, 1GB)"
        exit 1
    fi

    # Validate test type
    local valid_types="mixed put get delete stat"
    if [[ -n "$TEST_TYPE" ]] && [[ ! " $valid_types " =~ " $TEST_TYPE " ]]; then
        log_error "Invalid test type: $TEST_TYPE (valid: $valid_types)"
        exit 1
    fi

    log_success "Parameters validated"
}

# Prepare output directory
prepare_output() {
    log_info "Preparing output directory..."

    mkdir -p "$OUTPUT_DIR"

    # Create timestamped subdirectory
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    RESULT_DIR="$OUTPUT_DIR/test_$timestamp"
    mkdir -p "$RESULT_DIR"

    log_success "Results will be saved to: $RESULT_DIR"
}

# Generate test summary
generate_summary() {
    local result_file="$1"
    local summary_file="$RESULT_DIR/summary.txt"

    cat > "$summary_file" << EOF
S3 Stress Test Summary
======================
Date: $(date)
Host: $HOST
Bucket: $BUCKET
Test Type: ${TEST_TYPE:-mixed}
Concurrent: $CONCURRENT
Objects: $OBJECTS
Size: $SIZE
Duration: $DURATION
TLS: $TLS

Results saved to: $RESULT_DIR
EOF

    if [[ -f "$result_file" ]]; then
        echo "" >> "$summary_file"
        echo "Test Results:" >> "$summary_file"
        echo "=============" >> "$summary_file"

        # Extract key metrics from warp output
        grep -E "(Throughput|Bandwidth|Latency|Operations|Errors)" "$result_file" >> "$summary_file" 2>/dev/null || true
    fi

    log_success "Summary saved to: $summary_file"
}

# Run the actual benchmark
run_benchmark() {
    log_info "Starting S3 benchmark test..."
    log_info "Configuration:"
    log_info "  Host: $HOST"
    log_info "  Bucket: $BUCKET"
    log_info "  Test Type: ${TEST_TYPE:-mixed}"
    log_info "  Concurrent: $CONCURRENT"
    log_info "  Objects: $OBJECTS"
    log_info "  Size: $SIZE"
    log_info "  Duration: $DURATION"
    log_info "  TLS: $TLS"

    local result_file="$RESULT_DIR/warp_output.txt"
    local warp_cmd=(
        warp "${TEST_TYPE:-mixed}"
        --host "$HOST"
        --access-key "$ACCESS_KEY"
        --secret-key "$SECRET_KEY"
        --bucket "$BUCKET"
        --concurrent "$CONCURRENT"
        --objects "$OBJECTS"
        --obj.size "$SIZE"
        --duration "$DURATION"
    )

    # Add TLS flag if disabled
    [[ "$TLS" == "true" ]] || warp_cmd+=(--tls=false)

    # Add verbose flag if requested
    [[ "$VERBOSE" == "true" ]] && warp_cmd+=(--debug)

    log_info "Running: ${warp_cmd[*]}"

    # Run warp and capture output
    if "${warp_cmd[@]}" 2>&1 | tee "$result_file"; then
        log_success "Benchmark completed successfully"
        generate_summary "$result_file"
    else
        local exit_code=$?
        log_error "Benchmark failed with exit code: $exit_code"

        # Save error information
        echo "Exit code: $exit_code" >> "$result_file"
        generate_summary "$result_file"

        return $exit_code
    fi
}

# Cleanup function for trap
cleanup_on_exit() {
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        log_warning "Script interrupted (exit code: $exit_code)"
    fi

    # Cleanup bucket if not retained and not cleanup-only mode
    if [[ "$RETAIN_BUCKET" != "true" ]] && [[ "$CLEANUP_ONLY" != "true" ]]; then
        log_info "Cleaning up test bucket..."
        warp delete \
            --host "$HOST" \
            --access-key "$ACCESS_KEY" \
            --secret-key "$SECRET_KEY" \
            $([ "$TLS" == "true" ] || echo "--tls=false") \
            --bucket "$BUCKET" \
            2>/dev/null || true
    fi

    exit $exit_code
}

# Main function
main() {
    # Initialize variables with defaults
    HOST="$DEFAULT_HOST"
    ACCESS_KEY="$DEFAULT_ACCESS_KEY"
    SECRET_KEY="$DEFAULT_SECRET_KEY"
    BUCKET="$DEFAULT_BUCKET"
    CONCURRENT="$DEFAULT_CONCURRENT"
    OBJECTS="$DEFAULT_OBJECTS"
    SIZE="$DEFAULT_SIZE"
    DURATION="$DEFAULT_DURATION"
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    TLS="$DEFAULT_TLS"
    TEST_TYPE=""
    RETAIN_BUCKET=false
    VERBOSE=false
    CLEANUP_ONLY=false

    parse_args "$@"

    trap cleanup_on_exit EXIT INT TERM

    log_info "Starting S3 stress test script..."

    check_requirements
    install_warp

    if [[ "$CLEANUP_ONLY" == "true" ]]; then
        cleanup_buckets
        log_success "Cleanup completed successfully"
        exit 0
    fi

    test_connectivity
    validate_parameters
    prepare_output
    run_benchmark

    log_success "S3 stress test completed successfully!"
    log_info "Results available in: $RESULT_DIR"
}

# Run main function with all arguments
main "$@"