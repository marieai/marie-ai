#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_NAME="marie"
COMPOSE_FILES=(
    "./Dockerfiles/docker-compose.storage.yml"
    "./Dockerfiles/docker-compose.s3.yml"
    "./Dockerfiles/docker-compose.rabbitmq.yml"
    "./Dockerfiles/docker-compose.etcd.yml"
    "./Dockerfiles/docker-compose.gateway.yml"
    "./Dockerfiles/docker-compose.extract.yml"
    "./Dockerfiles/docker-compose.litellm.yml"
)

COMPOSE_ADDITIONAL_FILES="${COMPOSE_ADDITIONAL_FILES:-}"  # may be overwritten by CLI flag

# ENV_FILE can come from env or CLI; default path below
ENV_FILE="${ENV_FILE:-./config/.env.dev}"

DEPLOY_GATEWAY=${DEPLOY_GATEWAY:-true}
DEPLOY_EXTRACT=${DEPLOY_EXTRACT:-true}
DEPLOY_INFRASTRUCTURE=${DEPLOY_INFRASTRUCTURE:-true}
DEPLOY_LITELLM=${DEPLOY_LITELLM:-false}
DEPLOY_CLICKHOUSE=${DEPLOY_CLICKHOUSE:-true}
DEPLOY_GITEA=${DEPLOY_GITEA:-true}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Marie-AI System Bootstrap${NC}"
echo -e "${BLUE}========================================${NC}"

# Append any extra compose files provided via ENV/CLI
append_additional_compose_files() {
    IFS=',' read -ra ADDITIONAL_FILES <<< "$COMPOSE_ADDITIONAL_FILES"
    for file in "${ADDITIONAL_FILES[@]}"; do
        if [ -n "$file" ] && [ -f "$file" ]; then
            COMPOSE_FILES+=("$file")
            echo -e "${GREEN}✔ Added additional compose file: $file${NC}"
        elif [ -n "$file" ]; then
            echo -e "${YELLOW}⚠ Missing additional compose file: $file (skipped)${NC}"
        fi
    done
}

# Function to handle orphan removal based on environment
get_orphan_flag() {
    if [ "${COMPOSE_IGNORE_ORPHANS:-}" = "true" ]; then
        echo ""
    else
        echo "--remove-orphans"
    fi
}

get_running_services() {
    local service_type="$1"  # "infrastructure" or "application" or "all"

    case "$service_type" in
        "infrastructure")
            # Only check for infrastructure containers
            docker ps --format "table {{.Names}}\t{{.Status}}" --filter "name=marie-s3-server" --filter "name=marie-psql-server" --filter "name=marie-rabbitmq" --filter "name=etcd-single" --filter "name=marie-litellm" --filter "name=marie-mc-setup" --filter "name=marie-clickhouse" --filter "name=marie-gitea" 2>/dev/null | tail -n +2
            ;;
        "application")
            # Only check for application containers (gateway, extract, etc.)
            docker ps --format "table {{.Names}}\t{{.Status}}" --filter "name=marie-gateway" --filter "name=marie-extract" 2>/dev/null | tail -n +2
            ;;
        "all"|*)
            # Check for all Marie containers
            docker ps --format "table {{.Names}}\t{{.Status}}" --filter "name=${PROJECT_NAME}" 2>/dev/null | tail -n +2
            ;;
    esac
}

check_running_services() {
    local service_type="all"

    # Determine which services to check based on deployment configuration
    if [ "$DEPLOY_INFRASTRUCTURE" = "true" ] && ([ "$DEPLOY_GATEWAY" = "true" ] || [ "$DEPLOY_EXTRACT" = "true" ]); then
        service_type="all"
    elif [ "$DEPLOY_INFRASTRUCTURE" = "true" ]; then
        service_type="infrastructure"
    elif [ "$DEPLOY_GATEWAY" = "true" ] || [ "$DEPLOY_EXTRACT" = "true" ]; then
        service_type="application"
    fi

    local running_services
    running_services=$(get_running_services "$service_type")

    if [ -n "$running_services" ]; then
        echo -e "${YELLOW}Found running Marie-AI containers:${NC}"
        echo -e "${BLUE}NAMES               STATUS${NC}"
        echo "$running_services"
        echo ""
        echo -e "${YELLOW}⚠️  Warning: Running services detected!${NC}"

        if [ "$service_type" = "infrastructure" ]; then
            echo "To ensure a clean bootstrap, existing infrastructure services should be stopped."
        elif [ "$service_type" = "application" ]; then
            echo "To ensure a clean bootstrap, existing application services should be stopped."
        else
            echo "To ensure a clean bootstrap, existing services should be stopped."
        fi

        echo ""
        echo "Options:"
        if [ "$service_type" = "infrastructure" ]; then
            echo "1) Stop and remove infrastructure services (recommended)"
            echo "2) Stop infrastructure compose services only"
        elif [ "$service_type" = "application" ]; then
            echo "1) Stop and remove application services (recommended)"
            echo "2) Stop application compose services only"
        else
            echo "1) Stop and remove all Marie-AI containers (recommended)"
            echo "2) Stop compose services only"
        fi
        echo "3) Continue without cleanup (may cause conflicts)"
        echo "4) Exit"
        echo ""

        local choice
        read -p "Choose an option (1-4): " choice

        case $choice in
            1)
                if [ "$service_type" = "infrastructure" ]; then
                    stop_infrastructure_services
                elif [ "$service_type" = "application" ]; then
                    stop_application_services
                else
                    stop_all_services
                fi
                ;;
            2)
                if [ "$service_type" = "infrastructure" ]; then
                    stop_infrastructure_compose_services
                elif [ "$service_type" = "application" ]; then
                    stop_application_compose_services
                else
                    stop_all_compose_services
                fi
                ;;
            3)
                echo -e "${YELLOW}Continuing with existing services...${NC}"
                ;;
            4)
                echo -e "${BLUE}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Exiting...${NC}"
                exit 1
                ;;
        esac
        echo ""
    fi
}

stop_infrastructure_services() {
    echo -e "${YELLOW}Stopping infrastructure services...${NC}"

    local orphan_flag
    orphan_flag=$(get_orphan_flag)

    # Stop infrastructure services
    echo -e "${BLUE}🔧 Stopping infrastructure services...${NC}"
    COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE \
        --project-name marie-infrastructure \
        -f ./Dockerfiles/docker-compose.storage.yml \
        -f ./Dockerfiles/docker-compose.s3.yml \
        -f ./Dockerfiles/docker-compose.rabbitmq.yml \
        -f ./Dockerfiles/docker-compose.etcd.yml \
        -f ./Dockerfiles/docker-compose.litellm.yml \
        -f ./Dockerfiles/docker-compose.clickhouse.yml \
        -f ./Dockerfiles/docker-compose.gitea.yml \
        --project-directory . \
        down --volumes $orphan_flag 2>/dev/null || echo "No infrastructure services to stop"

    # Stop any remaining infrastructure containers
    local containers
    containers=$(docker ps -q --filter "name=marie-s3-server" --filter "name=marie-psql-server" --filter "name=marie-rabbitmq" --filter "name=etcd-single" --filter "name=marie-litellm" --filter "name=marie-mc-setup" --filter "name=marie-clickhouse" --filter "name=marie-gitea" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        echo "Stopping remaining infrastructure containers..."
        docker stop $containers 2>/dev/null || true
        docker rm $containers 2>/dev/null || true
    fi

    echo -e "${GREEN}✅ Infrastructure services stopped!${NC}"
}

stop_application_services() {
    echo -e "${YELLOW}Stopping application services...${NC}"

    local orphan_flag
    orphan_flag=$(get_orphan_flag)

    # Stop application services
    echo -e "${BLUE}🚀 Stopping application services...${NC}"
    COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE \
        --project-name marie-application \
        -f ./Dockerfiles/docker-compose.gateway.yml \
        -f ./Dockerfiles/docker-compose.extract.yml \
        --project-directory . \
        down --volumes $orphan_flag 2>/dev/null || echo "No application services to stop"

    # Stop any remaining application containers
    local containers
    containers=$(docker ps -q --filter "name=marie-gateway" --filter "name=marie-extract" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        echo "Stopping remaining application containers..."
        docker stop $containers 2>/dev/null || true
        docker rm $containers 2>/dev/null || true
    fi

    echo -e "${GREEN}✅ Application services stopped!${NC}"
}

stop_infrastructure_compose_services() {
    echo -e "${YELLOW}Stopping infrastructure compose services...${NC}"

    local orphan_flag
    orphan_flag=$(get_orphan_flag)

    COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE \
        --project-name marie-infrastructure \
        -f ./Dockerfiles/docker-compose.storage.yml \
        -f ./Dockerfiles/docker-compose.s3.yml \
        -f ./Dockerfiles/docker-compose.rabbitmq.yml \
        -f ./Dockerfiles/docker-compose.etcd.yml \
        -f ./Dockerfiles/docker-compose.litellm.yml \
        -f ./Dockerfiles/docker-compose.clickhouse.yml \
        -f ./Dockerfiles/docker-compose.gitea.yml \
        --project-directory . \
        down $orphan_flag 2>/dev/null || echo "No infrastructure services to stop"

    echo -e "${GREEN}✅ Infrastructure compose services stopped!${NC}"
}

stop_application_compose_services() {
    echo -e "${YELLOW}Stopping application compose services...${NC}"

    local orphan_flag
    orphan_flag=$(get_orphan_flag)

    COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE \
        --project-name marie-application \
        -f ./Dockerfiles/docker-compose.gateway.yml \
        -f ./Dockerfiles/docker-compose.extract.yml \
        --project-directory . \
        down $orphan_flag 2>/dev/null || echo "No application services to stop"

    echo -e "${GREEN}✅ Application compose services stopped!${NC}"
}

stop_all_compose_services() {
    echo -e "${YELLOW}Stopping all compose services...${NC}"
    stop_infrastructure_compose_services
    stop_application_compose_services
}

stop_all_services() {
    echo -e "${YELLOW}Stopping all Marie-AI services...${NC}"
    echo ""

    local orphan_flag
    orphan_flag=$(get_orphan_flag)

    # Stop infrastructure services
    echo -e "${BLUE}🔧 Stopping infrastructure services...${NC}"
    COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE \
        --project-name marie-infrastructure \
        -f ./Dockerfiles/docker-compose.storage.yml \
        -f ./Dockerfiles/docker-compose.s3.yml \
        -f ./Dockerfiles/docker-compose.rabbitmq.yml \
        -f ./Dockerfiles/docker-compose.etcd.yml \
        -f ./Dockerfiles/docker-compose.litellm.yml \
        -f ./Dockerfiles/docker-compose.clickhouse.yml \
        -f ./Dockerfiles/docker-compose.gitea.yml \
        --project-directory . \
        down --volumes $orphan_flag 2>/dev/null || echo "No infrastructure services to stop"

    # Stop application services
    echo -e "${BLUE}🚀 Stopping application services...${NC}"
    COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE \
        --project-name marie-application \
        -f ./Dockerfiles/docker-compose.gateway.yml \
        -f ./Dockerfiles/docker-compose.extract.yml \
        --project-directory . \
        down --volumes $orphan_flag 2>/dev/null || echo "No application services to stop"

    # Stop any remaining Marie containers
    echo -e "${BLUE}🧹 Cleaning up remaining containers...${NC}"
    local containers
    containers=$(docker ps -q --filter "name=${PROJECT_NAME}" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        echo "Stopping remaining containers..."
        docker stop $containers 2>/dev/null || true
    fi

    containers=$(docker ps -aq --filter "name=${PROJECT_NAME}" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        echo "Removing remaining containers..."
        docker rm $containers 2>/dev/null || true
    fi

    # Clean up unused volumes
    echo -e "${BLUE}💾 Cleaning up unused volumes...${NC}"
    docker volume prune -f 2>/dev/null || true

    echo ""
    echo -e "${GREEN}✅ All Marie-AI services stopped and cleaned up!${NC}"
    echo ""
}

show_deployment_config() {
    echo -e "${BLUE}Deployment Configuration:${NC}"
    echo -e "  Infrastructure: ${DEPLOY_INFRASTRUCTURE}"
    echo -e "    ├── Storage (MinIO): ${DEPLOY_INFRASTRUCTURE}"
    echo -e "    ├── Message Queue (RabbitMQ): ${DEPLOY_INFRASTRUCTURE}"
    echo -e "    ├── Service Discovery (etcd): ${DEPLOY_INFRASTRUCTURE}"
    echo -e "    ├── LLM Proxy (LiteLLM): ${DEPLOY_LITELLM}"
    echo -e "    ├── Analytics DB (ClickHouse): ${DEPLOY_CLICKHOUSE}"
    echo -e "    └── Git Service (Gitea): ${DEPLOY_GITEA}"
    echo -e "  Application Services:"
    echo -e "    ├── Gateway: ${DEPLOY_GATEWAY}"
    echo -e "    └── Extract Executors: ${DEPLOY_EXTRACT}"
    echo ""
}

validate_environment() {
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${RED}❌ Environment file not found: $ENV_FILE${NC}"
        echo "Please ensure the environment file exists before running bootstrap."
        exit 1
    fi
    echo -e "${GREEN}✅ Environment file found: $ENV_FILE${NC}"
}

create_docker_network() {
    local network_name="marie_default"
    if ! docker network inspect "$network_name" >/dev/null 2>&1; then
        echo -e "${BLUE}🔗 Creating Docker network: $network_name${NC}"
        docker network create --driver=bridge "$network_name"
        echo -e "${GREEN}✅ Network '$network_name' created${NC}"
    else
        echo -e "${GREEN}✅ Network '$network_name' already exists${NC}"
    fi
}

setup_gitea_admin() {
    local admin_user="${GITEA_ADMIN_USER:-marie}"
    local admin_password="${GITEA_ADMIN_PASSWORD:-rycerz}"
    local admin_email="${GITEA_ADMIN_EMAIL:-marie@marie.local}"

    echo -e "${BLUE}👤 Setting up Gitea admin user...${NC}"

    # Check if admin user already exists (run as git user)
    if docker exec -u git marie-gitea gitea admin user list 2>/dev/null | grep -q "$admin_user"; then
        echo -e "${GREEN}  ✅ Admin user '$admin_user' already exists${NC}"
        return 0
    fi

    # Create admin user (run as git user)
    if docker exec -u git marie-gitea gitea admin user create \
        --admin \
        --username "$admin_user" \
        --password "$admin_password" \
        --email "$admin_email" \
        --must-change-password=false 2>&1; then
        echo -e "${GREEN}  ✅ Admin user '$admin_user' created${NC}"
        echo -e "${YELLOW}     Username: $admin_user${NC}"
        echo -e "${YELLOW}     Password: $admin_password${NC}"
    else
        echo -e "${YELLOW}  ⚠️  Could not create admin user (may already exist)${NC}"
    fi
}

setup_gitea_oauth_app() {
    local admin_user="${GITEA_ADMIN_USER:-marie}"
    local admin_password="${GITEA_ADMIN_PASSWORD:-rycerz}"
    local gitea_url="http://localhost:${GITEA_HTTP_PORT:-3001}"
    local app_name="${GITEA_OAUTH_APP_NAME:-marie-studio}"
    local redirect_uri="${GITEA_OAUTH_REDIRECT_URI:-http://localhost:5173/api/auth/gitea/connect/callback}"

    echo -e "${BLUE}🔐 Setting up Gitea OAuth2 application...${NC}"

    # Check if OAuth app already exists
    local existing_apps
    existing_apps=$(curl -s -u "$admin_user:$admin_password" \
        "$gitea_url/api/v1/user/applications/oauth2")

    if echo "$existing_apps" | grep -q "\"name\":\"$app_name\""; then
        echo -e "${GREEN}  ✅ OAuth2 app '$app_name' already exists${NC}"
        # Extract and display existing credentials
        local client_id
        client_id=$(echo "$existing_apps" | grep -o "\"client_id\":\"[^\"]*\"" | head -1 | cut -d'"' -f4)
        if [ -n "$client_id" ]; then
            echo -e "${YELLOW}     Client ID: $client_id${NC}"
            echo -e "${YELLOW}     (Client Secret was shown only at creation time)${NC}"
        fi
        return 0
    fi

    # Create OAuth2 application
    local create_response
    create_response=$(curl -s -X POST \
        -u "$admin_user:$admin_password" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$app_name\", \"redirect_uris\": [\"$redirect_uri\"], \"confidential_client\": true}" \
        "$gitea_url/api/v1/user/applications/oauth2")

    local client_id
    local client_secret
    client_id=$(echo "$create_response" | grep -o '"client_id":"[^"]*"' | cut -d'"' -f4)
    client_secret=$(echo "$create_response" | grep -o '"client_secret":"[^"]*"' | cut -d'"' -f4)

    if [ -n "$client_id" ] && [ -n "$client_secret" ]; then
        echo -e "${GREEN}  ✅ OAuth2 app '$app_name' created${NC}"
        echo -e "${YELLOW}     ┌─────────────────────────────────────────────────────────┐${NC}"
        echo -e "${YELLOW}     │ SAVE THESE CREDENTIALS - Secret shown only once!        │${NC}"
        echo -e "${YELLOW}     ├─────────────────────────────────────────────────────────┤${NC}"
        echo -e "${YELLOW}     │ Gitea URL:     $gitea_url${NC}"
        echo -e "${YELLOW}     │ Client ID:     $client_id${NC}"
        echo -e "${YELLOW}     │ Client Secret: $client_secret${NC}"
        echo -e "${YELLOW}     │ Redirect URI:  $redirect_uri${NC}"
        echo -e "${YELLOW}     └─────────────────────────────────────────────────────────┘${NC}"
    else
        echo -e "${YELLOW}  ⚠️  Could not create OAuth2 app (may already exist)${NC}"
    fi
}

setup_gitea_repos() {
    local admin_user="${GITEA_ADMIN_USER:-marie}"
    local admin_password="${GITEA_ADMIN_PASSWORD:-rycerz}"
    local gitea_url="http://localhost:${GITEA_HTTP_PORT:-3001}"
    local repos="${GITEA_DEFAULT_REPOS:-planners:Query planners,prompts:Prompt templates,deployments:Deployment configs}"

    echo -e "${BLUE}📦 Setting up Gitea repositories...${NC}"

    # Parse and create each repository
    IFS=',' read -ra REPO_LIST <<< "$repos"
    for repo_entry in "${REPO_LIST[@]}"; do
        local repo_name="${repo_entry%%:*}"
        local repo_desc="${repo_entry#*:}"

        # Check if repo exists using API
        local check_response
        check_response=$(curl -s -o /dev/null -w "%{http_code}" \
            -u "$admin_user:$admin_password" \
            "$gitea_url/api/v1/repos/$admin_user/$repo_name")

        if [ "$check_response" = "200" ]; then
            echo -e "${GREEN}  ✅ Repository '$repo_name' already exists${NC}"
            continue
        fi

        # Create repository using API
        local create_response
        create_response=$(curl -s -w "\n%{http_code}" \
            -X POST \
            -u "$admin_user:$admin_password" \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"$repo_name\", \"description\": \"$repo_desc\", \"private\": false, \"auto_init\": true}" \
            "$gitea_url/api/v1/user/repos")

        local http_code
        http_code=$(echo "$create_response" | tail -n1)

        if [ "$http_code" = "201" ]; then
            echo -e "${GREEN}  ✅ Repository '$repo_name' created${NC}"
        else
            echo -e "${YELLOW}  ⚠️  Could not create repository '$repo_name' (may already exist)${NC}"
        fi
    done
}

initialize_databases() {
    echo -e "${BLUE}🗄️  Initializing databases...${NC}"

    # Wait for PostgreSQL to accept connections
    local max_attempts=30
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if docker exec marie-psql-server pg_isready -U "${POSTGRES_USER:-postgres}" >/dev/null 2>&1; then
            break
        fi
        echo "  Waiting for PostgreSQL to be ready (attempt $attempt/$max_attempts)..."
        sleep 2
        ((attempt++))
    done

    if [ $attempt -gt $max_attempts ]; then
        echo -e "${RED}❌ PostgreSQL did not become ready in time${NC}"
        return 1
    fi

    # Create Gitea database if it doesn't exist
    if [ "$DEPLOY_GITEA" = "true" ]; then
        local gitea_db="${GITEA_DB_NAME:-gitea}"
        echo "  Creating database '$gitea_db' for Gitea..."
        docker exec marie-psql-server psql -U "${POSTGRES_USER:-postgres}" -tc \
            "SELECT 1 FROM pg_database WHERE datname = '$gitea_db'" | grep -q 1 || \
            docker exec marie-psql-server psql -U "${POSTGRES_USER:-postgres}" -c \
            "CREATE DATABASE $gitea_db" >/dev/null 2>&1
        echo -e "${GREEN}  ✅ Database '$gitea_db' ready${NC}"
    fi

    # Create LiteLLM database if needed (for future use)
    if [ "$DEPLOY_LITELLM" = "true" ]; then
        echo "  Creating database 'litellm' for LiteLLM..."
        docker exec marie-psql-server psql -U "${POSTGRES_USER:-postgres}" -tc \
            "SELECT 1 FROM pg_database WHERE datname = 'litellm'" | grep -q 1 || \
            docker exec marie-psql-server psql -U "${POSTGRES_USER:-postgres}" -c \
            "CREATE DATABASE litellm" >/dev/null 2>&1
        echo -e "${GREEN}  ✅ Database 'litellm' ready${NC}"
    fi

    # Initialize ClickHouse databases
    if [ "$DEPLOY_CLICKHOUSE" = "true" ]; then
        echo "  Initializing ClickHouse databases..."

        # Wait for ClickHouse to be ready
        local ch_attempts=20
        local ch_attempt=1
        while [ $ch_attempt -le $ch_attempts ]; do
            if docker exec marie-clickhouse clickhouse-client --query "SELECT 1" >/dev/null 2>&1; then
                break
            fi
            echo "  Waiting for ClickHouse to be ready (attempt $ch_attempt/$ch_attempts)..."
            sleep 2
            ((ch_attempt++))
        done

        if [ $ch_attempt -gt $ch_attempts ]; then
            echo -e "${YELLOW}⚠️  ClickHouse not ready, skipping database creation${NC}"
        else
            # Create marie database for LLM tracking and analytics
            local ch_db="${CLICKHOUSE_DB:-marie}"
            echo "  Creating ClickHouse database '$ch_db'..."
            docker exec marie-clickhouse clickhouse-client --query \
                "CREATE DATABASE IF NOT EXISTS $ch_db" >/dev/null 2>&1
            echo -e "${GREEN}  ✅ ClickHouse database '$ch_db' ready${NC}"
        fi
    fi

    echo -e "${GREEN}✅ Database initialization complete${NC}"
}

validate_compose_files() {
    local missing_files=()
    local optional_files=(
        "./Dockerfiles/docker-compose.gateway.yml"
        "./Dockerfiles/docker-compose.extract.yml"
        "./Dockerfiles/docker-compose.litellm.yml"
    )

    for compose_file in "${COMPOSE_FILES[@]}"; do
        if [ ! -f "$compose_file" ]; then
            # Check if it's an optional file based on deployment flags
            local is_optional=false
            for optional_file in "${optional_files[@]}"; do
                if [ "$compose_file" = "$optional_file" ]; then
                    is_optional=true
                    break
                fi
            done

            if [ "$is_optional" = false ]; then
                missing_files+=("$compose_file")
            else
                echo -e "${YELLOW}⚠️  Optional file missing: $compose_file${NC}"
            fi
        fi
    done

    if [ ${#missing_files[@]} -gt 0 ]; then
        echo -e "${RED}❌ Missing required compose files:${NC}"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi

    echo -e "${GREEN}✅ All required compose files found.${NC}"
}

bootstrap_system() {
    echo ""
    echo -e "${BLUE}Starting Marie-AI system bootstrap...${NC}"

    source "$ENV_FILE"
    echo -e "${GREEN}✅ Environment loaded from $ENV_FILE${NC}"

    # Create Docker network if it doesn't exist
    create_docker_network

    local orphan_flag
    orphan_flag=$(get_orphan_flag)

    # Stage 1: Start infrastructure services with separate project name
    if [ "$DEPLOY_INFRASTRUCTURE" = "true" ]; then
        echo -e "${BLUE}🔧 Stage 1: Starting infrastructure services...${NC}"

        local infra_compose_cmd="docker compose --env-file $ENV_FILE"
        infra_compose_cmd="$infra_compose_cmd --project-name marie-infrastructure"
        infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.storage.yml"
        infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.s3.yml"
        infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
        infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.etcd.yml"

        # Note: LiteLLM is NOT included here - it will be started after database initialization
        # because it requires the 'litellm' database to exist before starting

        if [ "$DEPLOY_CLICKHOUSE" = "true" ] && [ -f "./Dockerfiles/docker-compose.clickhouse.yml" ]; then
            infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.clickhouse.yml"
        fi

        # Note: Gitea is NOT included here - it will be started after database initialization
        # because it requires the 'gitea' database to exist before starting

        infra_compose_cmd="$infra_compose_cmd --project-directory ."

        echo "Starting infrastructure services with host networking..."
        # Use host networking for all services
        COMPOSE_NETWORK_MODE=host eval "$infra_compose_cmd up -d --build $orphan_flag"

        echo -e "${YELLOW}⏳ Waiting for core infrastructure services to be healthy...${NC}"

        # First wait for core services (PostgreSQL must be ready before we can create databases)
        # Note: LiteLLM and Gitea are NOT included - they need their databases created first
        local core_services_to_wait=("s3server" "psql" "rabbitmq" "etcd-single")

        if [ "$DEPLOY_CLICKHOUSE" = "true" ]; then
            core_services_to_wait+=("clickhouse")
        fi

        # Wait for core services first (excluding LiteLLM and Gitea which need DB setup)
        local wait_compose_cmd="COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE"
        wait_compose_cmd="$wait_compose_cmd --project-name marie-infrastructure"
        wait_compose_cmd="$wait_compose_cmd -f ./Dockerfiles/docker-compose.storage.yml"
        wait_compose_cmd="$wait_compose_cmd -f ./Dockerfiles/docker-compose.s3.yml"
        wait_compose_cmd="$wait_compose_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
        wait_compose_cmd="$wait_compose_cmd -f ./Dockerfiles/docker-compose.etcd.yml"

        if [ "$DEPLOY_CLICKHOUSE" = "true" ]; then
            wait_compose_cmd="$wait_compose_cmd -f ./Dockerfiles/docker-compose.clickhouse.yml"
        fi

        # Note: LiteLLM and Gitea are not included in wait - they haven't been started yet

        wait_compose_cmd="$wait_compose_cmd --project-directory . up --wait ${core_services_to_wait[*]}"
        eval "$wait_compose_cmd"

        # Initialize databases after PostgreSQL is ready
        initialize_databases

        # Now start and wait for LiteLLM (after database is created)
        # Important: Include all infra compose files to avoid orphan removal
        if [ "$DEPLOY_LITELLM" = "true" ] && [ -f "./Dockerfiles/docker-compose.litellm.yml" ]; then
            echo -e "${YELLOW}⏳ Starting LiteLLM (database is now ready)...${NC}"
            local litellm_cmd="COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE"
            litellm_cmd="$litellm_cmd --project-name marie-infrastructure"
            litellm_cmd="$litellm_cmd -f ./Dockerfiles/docker-compose.storage.yml"
            litellm_cmd="$litellm_cmd -f ./Dockerfiles/docker-compose.s3.yml"
            litellm_cmd="$litellm_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
            litellm_cmd="$litellm_cmd -f ./Dockerfiles/docker-compose.etcd.yml"
            if [ "$DEPLOY_CLICKHOUSE" = "true" ]; then
                litellm_cmd="$litellm_cmd -f ./Dockerfiles/docker-compose.clickhouse.yml"
            fi
            litellm_cmd="$litellm_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
            litellm_cmd="$litellm_cmd --project-directory ."
            # Start LiteLLM (no orphan flag to be safe)
            eval "$litellm_cmd up -d litellm"
            # Wait for it to be healthy
            echo -e "${YELLOW}⏳ Waiting for LiteLLM to be healthy...${NC}"
            eval "$litellm_cmd up --wait litellm"
            echo -e "${GREEN}✅ LiteLLM is ready${NC}"
        fi

        # Now start and wait for Gitea (after database is created)
        # Important: Include all infra compose files to avoid orphan removal
        if [ "$DEPLOY_GITEA" = "true" ] && [ -f "./Dockerfiles/docker-compose.gitea.yml" ]; then
            echo -e "${YELLOW}⏳ Starting Gitea (database is now ready)...${NC}"
            local gitea_cmd="COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE"
            gitea_cmd="$gitea_cmd --project-name marie-infrastructure"
            gitea_cmd="$gitea_cmd -f ./Dockerfiles/docker-compose.storage.yml"
            gitea_cmd="$gitea_cmd -f ./Dockerfiles/docker-compose.s3.yml"
            gitea_cmd="$gitea_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
            gitea_cmd="$gitea_cmd -f ./Dockerfiles/docker-compose.etcd.yml"
            if [ "$DEPLOY_CLICKHOUSE" = "true" ]; then
                gitea_cmd="$gitea_cmd -f ./Dockerfiles/docker-compose.clickhouse.yml"
            fi
            gitea_cmd="$gitea_cmd -f ./Dockerfiles/docker-compose.gitea.yml"
            gitea_cmd="$gitea_cmd --project-directory ."
            # Start Gitea (no orphan flag to be safe)
            eval "$gitea_cmd up -d gitea"
            # Wait for it to be healthy
            echo -e "${YELLOW}⏳ Waiting for Gitea to be healthy...${NC}"
            eval "$gitea_cmd up --wait gitea"
            echo -e "${GREEN}✅ Gitea is ready${NC}"

            # Create default admin user
            setup_gitea_admin

            # Create default repositories
            setup_gitea_repos

            # Create OAuth2 application for Marie Studio
            setup_gitea_oauth_app
        fi

        # Check if mc-setup completed successfully
        echo -e "${YELLOW}Checking MinIO setup completion...${NC}"
        local setup_attempts=30
        local setup_attempt=1

        while [ $setup_attempt -le $setup_attempts ]; do
            local setup_status
            setup_status=$(docker inspect marie-mc-setup --format='{{.State.Status}}' 2>/dev/null || echo "not_found")

            if [ "$setup_status" = "exited" ]; then
                local exit_code
                exit_code=$(docker inspect marie-mc-setup --format='{{.State.ExitCode}}' 2>/dev/null || echo "1")
                if [ "$exit_code" = "0" ]; then
                    echo -e "${GREEN}✅ MinIO setup completed successfully${NC}"
                    break
                else
                    echo -e "${RED}❌ MinIO setup failed with exit code $exit_code${NC}"
                    echo "Setup logs:"
                    docker logs marie-mc-setup --tail 20
                    exit 1
                fi
            elif [ "$setup_status" = "not_found" ]; then
                echo -e "${RED}❌ MinIO setup container not found${NC}"
                exit 1
            elif [ $setup_attempt -eq $setup_attempts ]; then
                echo -e "${RED}❌ MinIO setup did not complete within expected time${NC}"
                echo "Current status: $setup_status"
                docker logs marie-mc-setup --tail 20
                exit 1
            else
                echo "  Attempt $setup_attempt/$setup_attempts - MinIO setup status: $setup_status"
                sleep 2
                ((setup_attempt++))
            fi
        done

        echo -e "${GREEN}✅ Infrastructure services are ready${NC}"
    fi

    # Stage 2: Start application services with separate project name
    echo -e "${BLUE}🚀 Stage 2: Starting application services...${NC}"

    local app_compose_cmd="docker compose --env-file $ENV_FILE"
    app_compose_cmd="$app_compose_cmd --project-name marie-application"
    local has_app_services=false

    if [ "$DEPLOY_GATEWAY" = "true" ] && [ -f "./Dockerfiles/docker-compose.gateway.yml" ]; then
        app_compose_cmd="$app_compose_cmd -f ./Dockerfiles/docker-compose.gateway.yml"
        has_app_services=true
    fi

    if [ "$DEPLOY_EXTRACT" = "true" ] && [ -f "./Dockerfiles/docker-compose.extract.yml" ]; then
        app_compose_cmd="$app_compose_cmd -f ./Dockerfiles/docker-compose.extract.yml"
        has_app_services=true
    fi

    app_compose_cmd="$app_compose_cmd --project-directory ."

    if [ "$has_app_services" = true ]; then
        echo "Starting application services with host networking..."
        # Use host networking for application services too
        COMPOSE_NETWORK_MODE=host eval "$app_compose_cmd up -d --build $orphan_flag"
    else
        echo -e "${YELLOW}No application services configured to start${NC}"
    fi

    echo ""
    echo -e "${GREEN}🎉 Marie-AI system started successfully!${NC}"
    echo ""

    echo "Services status:"
    show_all_services_status
    show_service_endpoints
}

show_all_services_status() {
    if [ "$DEPLOY_INFRASTRUCTURE" = "true" ]; then
        echo -e "${BLUE}Infrastructure Services:${NC}"
        local status_compose_cmd="COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE"
        status_compose_cmd="$status_compose_cmd --project-name marie-infrastructure"
        status_compose_cmd="$status_compose_cmd -f ./Dockerfiles/docker-compose.storage.yml"
        status_compose_cmd="$status_compose_cmd -f ./Dockerfiles/docker-compose.s3.yml"
        status_compose_cmd="$status_compose_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
        status_compose_cmd="$status_compose_cmd -f ./Dockerfiles/docker-compose.etcd.yml"

        if [ "$DEPLOY_LITELLM" = "true" ]; then
            status_compose_cmd="$status_compose_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
        fi

        if [ "$DEPLOY_CLICKHOUSE" = "true" ]; then
            status_compose_cmd="$status_compose_cmd -f ./Dockerfiles/docker-compose.clickhouse.yml"
        fi

        if [ "$DEPLOY_GITEA" = "true" ]; then
            status_compose_cmd="$status_compose_cmd -f ./Dockerfiles/docker-compose.gitea.yml"
        fi

        status_compose_cmd="$status_compose_cmd --project-directory . ps"
        eval "$status_compose_cmd" 2>/dev/null || echo "No infrastructure services"
    fi

    if [ "$DEPLOY_GATEWAY" = "true" ] || [ "$DEPLOY_EXTRACT" = "true" ]; then
        echo ""
        echo -e "${BLUE}Application Services:${NC}"
        COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE \
            --project-name marie-application \
            -f ./Dockerfiles/docker-compose.gateway.yml \
            -f ./Dockerfiles/docker-compose.extract.yml \
            --project-directory . \
            ps 2>/dev/null || echo "No application services"
    fi
}

show_service_endpoints() {
    echo ""
    echo -e "${BLUE}🔗 Service Endpoints:${NC}"

    if [ "$DEPLOY_INFRASTRUCTURE" = "true" ]; then
        echo -e "${GREEN}Infrastructure Services:${NC}"
        echo "  🐰 RabbitMQ Management: http://localhost:15672 (${RABBIT_MQ_USERNAME}/${RABBIT_MQ_PASSWORD})"
        echo "  💾 MinIO S3 API: http://localhost:9000 (marieadmin/marietopsecret)"
        echo "  💾 MinIO Console: http://localhost:9001 (marieadmin/marietopsecret)"
        echo "  🗄️  etcd: http://localhost:2379"

        if [ "$DEPLOY_LITELLM" = "true" ]; then
            echo "  🤖 LiteLLM Proxy: http://localhost:4000"
            echo "  📊 LiteLLM Admin UI: http://localhost:4000/ui"
            echo "  🔧 LiteLLM Health: http://localhost:4000/health"
        fi

        if [ "$DEPLOY_CLICKHOUSE" = "true" ]; then
            echo "  📈 ClickHouse HTTP: http://localhost:8123"
            echo "  📈 ClickHouse Play: http://localhost:8123/play"
            echo "  📈 ClickHouse Native: localhost:9000"
        fi

        if [ "$DEPLOY_GITEA" = "true" ]; then
            echo "  🐙 Gitea Web UI: http://localhost:3001 (${GITEA_ADMIN_USER:-marie}/${GITEA_ADMIN_PASSWORD:-rycerz})"
            echo "  🐙 Gitea SSH: ssh://git@localhost:2222"
        fi
    fi

    if [ "$DEPLOY_GATEWAY" = "true" ]; then
        echo -e "${GREEN}Application Services:${NC}"
        echo "  🌐 HTTP Gateway: http://localhost:52000"
        echo "  🔌 GRPC Gateway: grpc://localhost:51000"
    fi

    if [ "$DEPLOY_EXTRACT" = "true" ]; then
        echo "  🔍 Extract Executor: http://localhost:8080"
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env-file)
                ENV_FILE="$2"
                shift 2
                ;;
            --additional-files)
                COMPOSE_ADDITIONAL_FILES="$2"
                shift 2
                ;;
            --stop-all)
                stop_all_services
                exit 0
                ;;
            --no-gateway)
                DEPLOY_GATEWAY=false
                shift
                ;;
            --no-extract)
                DEPLOY_EXTRACT=false
                shift
                ;;
            --no-infrastructure)
                DEPLOY_INFRASTRUCTURE=false
                DEPLOY_LITELLM=false  # LiteLLM is part of infrastructure
                shift
                ;;
            --no-litellm)
                DEPLOY_LITELLM=false
                shift
                ;;
            --no-clickhouse)
                DEPLOY_CLICKHOUSE=false
                shift
                ;;
            --no-gitea)
                DEPLOY_GITEA=false
                shift
                ;;
            --infrastructure-only)
                DEPLOY_GATEWAY=false
                DEPLOY_EXTRACT=false
                # Keep DEPLOY_LITELLM as is (part of infrastructure)
                shift
                ;;
            --services-only)
                DEPLOY_INFRASTRUCTURE=false
                DEPLOY_LITELLM=false  # LiteLLM is part of infrastructure
                shift
                ;;
            --litellm-only)
                DEPLOY_GATEWAY=false
                DEPLOY_EXTRACT=false
                DEPLOY_INFRASTRUCTURE=true  # Need infrastructure for LiteLLM
                DEPLOY_LITELLM=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --env-file PATH       Path to .env file (default: ./config/.env.dev)"
    echo "  --additional-files    FILE1.yml[,FILE2.yml]  Extra docker-compose files to include"
    echo "  --stop-all            Stop and remove all Marie-AI services and containers"
    echo "  --no-gateway          Skip gateway deployment"
    echo "  --no-extract          Skip extract executor deployment"
    echo "  --no-infrastructure   Skip infrastructure services (includes LiteLLM, ClickHouse, Gitea)"
    echo "  --no-litellm          Skip LiteLLM proxy deployment"
    echo "  --no-clickhouse       Skip ClickHouse analytics database deployment"
    echo "  --no-gitea            Skip Gitea Git service deployment"
    echo "  --infrastructure-only Deploy only infrastructure services"
    echo "  --services-only       Deploy only Marie application services (gateway + extract)"
    echo "  --litellm-only        Deploy only LiteLLM proxy (with required infrastructure)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Service Categories:"
    echo "  Infrastructure: Storage (MinIO), Message Queue (RabbitMQ), Service Discovery (etcd),"
    echo "                  LLM Proxy (LiteLLM), Analytics DB (ClickHouse), Git Service (Gitea)"
    echo "  Application:    Gateway, Extract Executors"
    echo ""
    echo "Examples:"
    echo "  $0                        # Deploy everything"
    echo "  $0 --stop-all             # Stop all services and cleanup"
    echo "  $0 --infrastructure-only  # Deploy infrastructure only"
    echo "  $0 --services-only        # Deploy only gateway + extract"
    echo "  $0 --no-extract           # Deploy infrastructure + gateway only"
    echo "  $0 --no-clickhouse --no-gitea  # Deploy without analytics and Git"
    echo "  $0 --litellm-only         # Deploy minimal infrastructure + LiteLLM"
}

main() {
    parse_args "$@"

    append_additional_compose_files
    show_deployment_config
    validate_environment
    validate_compose_files

    check_running_services

    bootstrap_system

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Bootstrap completed successfully!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Useful commands:"
    echo "  Stop all services: ./bootstrap-marie.sh --stop-all"
    echo "  View logs: docker compose logs -f [service_name]"
    echo "  Stop services: docker compose down"
    echo "  Stop and cleanup: docker compose down --volumes --remove-orphans"
    if [ "$DEPLOY_LITELLM" = "true" ]; then
        echo "  View LiteLLM logs: docker compose logs -f litellm"
        echo "  LiteLLM health check: curl http://localhost:4000/health"
    fi
}

main "$@"