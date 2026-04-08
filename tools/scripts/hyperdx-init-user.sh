#!/bin/bash
# ============================================================================
# HyperDX Initial User Setup
# ============================================================================
# Creates or resets the default admin user for HyperDX.
#
# Usage:
#   ./tools/scripts/hyperdx-init-user.sh [--env-file <path>] [email] [password]
#
# Examples:
#   ./tools/scripts/hyperdx-init-user.sh --env-file ./config/.env.dev
#   ./tools/scripts/hyperdx-init-user.sh admin@example.com mypassword
#
# Environment Variables:
#   HYPERDX_ADMIN_EMAIL    - Admin email (default: admin@localhost)
#   HYPERDX_ADMIN_PASSWORD - Admin password (default: admin123)
#   CLICKHOUSE_HOST            - ClickHouse host for HyperDX (default: marie-clickhouse)
#   CLICKHOUSE_HTTP_PORT       - ClickHouse HTTP port (default: 8123)
#   CLICKHOUSE_APP_USER        - ClickHouse app user (default: marie)
#   CLICKHOUSE_APP_PASSWORD    - ClickHouse app password
#   CLICKHOUSE_APP_DATABASE    - ClickHouse app database (default: otel)
#
# Prerequisites:
#   - marie-hyperdx container must be running
#   - marie-ferretdb container must be running
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./tools/scripts/hyperdx-init-user.sh [--env-file <path>] [email] [password]

Examples:
  ./tools/scripts/hyperdx-init-user.sh --env-file ./config/.env.dev
  ./tools/scripts/hyperdx-init-user.sh admin@example.com mypassword
EOF
}

resolve_env_file() {
    local env_path="$1"

    if [[ -f "$env_path" ]]; then
        printf '%s\n' "$env_path"
        return 0
    fi

    if [[ -f "${REPO_ROOT}/${env_path}" ]]; then
        printf '%s\n' "${REPO_ROOT}/${env_path}"
        return 0
    fi

    return 1
}

# Parse arguments
ENV_FILE=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-file|-e)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --env-file requires a path argument" >&2
                usage
                exit 1
            fi
            ENV_FILE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Load env file if provided
if [[ -n "$ENV_FILE" ]]; then
    if ! RESOLVED_ENV_FILE="$(resolve_env_file "$ENV_FILE")"; then
        echo "ERROR: Env file not found: $ENV_FILE" >&2
        exit 1
    fi
    echo "==> Loading environment from: $RESOLVED_ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    source "$RESOLVED_ENV_FILE"
    set +a
fi

# Set variables from positional args or environment
ADMIN_EMAIL="${POSITIONAL_ARGS[0]:-${HYPERDX_ADMIN_EMAIL:-admin@localhost}}"
ADMIN_PASSWORD="${POSITIONAL_ARGS[1]:-${HYPERDX_ADMIN_PASSWORD:-admin123}}"
CONTAINER_NAME="${HYPERDX_CONTAINER:-marie-hyperdx}"
CLICKHOUSE_HOST="${CLICKHOUSE_HOST:-marie-clickhouse}"
CLICKHOUSE_PORT="${CLICKHOUSE_HTTP_PORT:-8123}"
CLICKHOUSE_USER="${CLICKHOUSE_APP_USER:-marie}"
CLICKHOUSE_PASSWORD="${CLICKHOUSE_APP_PASSWORD:-}"
CLICKHOUSE_DATABASE="${CLICKHOUSE_APP_DATABASE:-otel}"
MAX_RETRIES=30
RETRY_INTERVAL=2

echo "==> HyperDX User Setup"
echo "    Email: ${ADMIN_EMAIL}"
echo "    Container: ${CONTAINER_NAME}"

if [[ "$CLICKHOUSE_HOST" == "localhost" || "$CLICKHOUSE_HOST" == "127.0.0.1" ]]; then
    echo "WARNING: ClickHouse host is set to '${CLICKHOUSE_HOST}'."
    echo "         HyperDX runs in Docker, so localhost points to the HyperDX container."
    echo "         Use the Docker service/container hostname instead (default: marie-clickhouse)."
fi

# Wait for container to be running and API responding
echo "==> Waiting for ${CONTAINER_NAME} to be ready..."
for i in $(seq 1 $MAX_RETRIES); do
    # Check if container is running
    if ! docker inspect --format='{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "true"; then
        echo "    Container not running yet... ($i/$MAX_RETRIES)"
        sleep $RETRY_INTERVAL
        continue
    fi

    # Check if API is responding (wget returns 0 on success)
    if docker exec "$CONTAINER_NAME" wget -q --spider http://localhost:8080/ 2>/dev/null; then
        echo "    Container is ready"
        break
    fi

    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: Container did not become ready within $((MAX_RETRIES * RETRY_INTERVAL)) seconds"
        exit 1
    fi
    echo "    Waiting for API... ($i/$MAX_RETRIES)"
    sleep $RETRY_INTERVAL
done

# Additional wait for API to be fully ready
echo "==> Waiting for HyperDX API to initialize..."
sleep 3

# Create/reset user
echo "==> Creating/resetting admin user..."
docker exec -e ADMIN_EMAIL="$ADMIN_EMAIL" -e ADMIN_PASSWORD="$ADMIN_PASSWORD" "$CONTAINER_NAME" sh -c 'cd /app/api/packages/api/build && node -e "
const config = require(\"./config\");
const User = require(\"./models/user\").default;
const Team = require(\"./models/team\").default;
const mongoose = require(\"mongoose\");

const EMAIL = process.env.ADMIN_EMAIL;
const PASSWORD = process.env.ADMIN_PASSWORD;

(async () => {
  try {
    await mongoose.connect(config.MONGO_URI);

    // Check if user exists
    let user = await User.findOne({email: EMAIL});

    if (!user) {
      // Check for existing team or create one
      let team = await Team.findOne({});
      if (!team) {
        team = new Team({name: \"Default Team\"});
        await team.save();
        console.log(\"Created team: Default Team\");
      }

      // Create user
      user = new User({
        email: EMAIL,
        team: team._id,
        isVerified: true
      });
      await user.setPassword(PASSWORD);
      await user.save();

      // Add user to team
      team.users = team.users || [];
      if (!team.users.includes(user._id)) {
        team.users.push(user._id);
        await team.save();
      }

      console.log(\"Created admin user: \" + EMAIL);
    } else {
      await user.setPassword(PASSWORD);
      await user.save();
      console.log(\"Reset password for: \" + EMAIL);
    }

    await mongoose.disconnect();
    process.exit(0);
  } catch (err) {
    console.error(\"Error:\", err.message);
    process.exit(1);
  }
})();
"'

echo "==> Done! You can now login at http://localhost:8080"
echo "    Email: ${ADMIN_EMAIL}"
echo "    Password: (as configured)"
echo ""
echo "==> HyperDX ClickHouse connection settings"
echo "    Host: ${CLICKHOUSE_HOST}"
echo "    Port: ${CLICKHOUSE_PORT}"
echo "    User: ${CLICKHOUSE_USER}"
echo "    Database: ${CLICKHOUSE_DATABASE}"

if docker exec "$CONTAINER_NAME" sh -c "wget -q --spider http://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}/ping" >/dev/null 2>&1; then
    echo "    Reachability: OK (HyperDX container can reach ClickHouse)"
else
    echo "WARNING: HyperDX container could not reach ClickHouse at ${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}"
    echo "         If HyperDX shows 'Failed to connect to ClickHouse server', do not use localhost."
    echo "         Use the Docker hostname shown above."
fi
