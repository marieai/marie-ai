#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
ENV_FILE="./config/.env.dev"

# Service deployment options
DEPLOY_GATEWAY=${DEPLOY_GATEWAY:-true}
DEPLOY_EXTRACT=${DEPLOY_EXTRACT:-true}
DEPLOY_INFRASTRUCTURE=${DEPLOY_INFRASTRUCTURE:-true}
DEPLOY_LITELLM=${DEPLOY_LITELLM:-true}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Marie-AI System Bootstrap${NC}"
echo -e "${BLUE}========================================${NC}"

show_deployment_config() {
    echo -e "${BLUE}Deployment Configuration:${NC}"
    echo -e "  Infrastructure: ${DEPLOY_INFRASTRUCTURE}"
    echo -e "  Gateway: ${DEPLOY_GATEWAY}"
    echo -e "  Extract Executors: ${DEPLOY_EXTRACT}"
    echo -e "  LiteLLM Proxy: ${DEPLOY_LITELLM}"
    echo ""
}

check_running_services() {
    local running_containers
    running_containers=$(docker ps --filter "name=${PROJECT_NAME}" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || true)

    if [ -n "$running_containers" ] && [ "$(echo "$running_containers" | wc -l)" -gt 1 ]; then
        echo -e "${YELLOW}Found running Marie-AI containers:${NC}"
        echo "$running_containers"
        return 0
    else
        return 1
    fi
}

check_compose_services() {
    local services_exist=false

    for compose_file in "${COMPOSE_FILES[@]}"; do
        if [ -f "$compose_file" ]; then
            local services
            services=$(docker compose -f "$compose_file" ps --services 2>/dev/null || true)
            if [ -n "$services" ]; then
                local running_services
                running_services=$(docker compose -f "$compose_file" ps --filter "status=running" --format "table {{.Service}}\t{{.Status}}" 2>/dev/null || true)
                if [ -n "$running_services" ] && [ "$(echo "$running_services" | wc -l)" -gt 1 ]; then
                    if [ "$services_exist" = false ]; then
                        echo -e "${YELLOW}Found running compose services:${NC}"
                        services_exist=true
                    fi
                    echo -e "${YELLOW}From $compose_file:${NC}"
                    echo "$running_services"
                fi
            fi
        fi
    done

    if [ "$services_exist" = true ]; then
        return 0
    else
        return 1
    fi
}

prompt_cleanup() {
    echo ""
    echo -e "${RED}⚠️  Warning: Running services detected!${NC}"
    echo "To ensure a clean bootstrap, existing services should be stopped."
    echo ""
    echo "Options:"
    echo "1) Stop and remove all Marie-AI containers (recommended)"
    echo "2) Stop compose services only"
    echo "3) Continue without cleanup (may cause conflicts)"
    echo "4) Exit"
    echo ""

    while true; do
        read -p "Choose an option [1-4]: " choice
        case $choice in
            1)
                cleanup_all_containers
                break
                ;;
            2)
                cleanup_compose_services
                break
                ;;
            3)
                echo -e "${YELLOW}Continuing without cleanup...${NC}"
                break
                ;;
            4)
                echo -e "${RED}Bootstrap cancelled.${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Please choose 1-4.${NC}"
                ;;
        esac
    done
}

cleanup_all_containers() {
    echo -e "${YELLOW}Stopping and removing all Marie-AI containers...${NC}"

    local containers
    containers=$(docker ps -q --filter "name=${PROJECT_NAME}" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        echo "Stopping containers..."
        docker stop $containers
    fi

    containers=$(docker ps -aq --filter "name=${PROJECT_NAME}" 2>/dev/null || true)
    if [ -n "$containers" ]; then
        echo "Removing containers..."
        docker rm $containers
    fi

    if docker network ls | grep -q marie_default; then
        echo "Checking marie_default network..."
        local network_containers
        network_containers=$(docker network inspect marie_default --format='{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || true)
        if [ -z "$network_containers" ]; then
            echo "Removing unused marie_default network..."
            docker network rm marie_default 2>/dev/null || true
        fi
    fi

    echo -e "${GREEN}  Container cleanup completed.${NC}"
}

cleanup_compose_services() {
    echo -e "${YELLOW}Stopping Docker Compose services...${NC}"

    for compose_file in "${COMPOSE_FILES[@]}"; do
        if [ -f "$compose_file" ]; then
            echo "Processing $compose_file..."
            docker compose \
                --env-file "$ENV_FILE" \
                -f "$compose_file" \
                --project-directory . \
                down --remove-orphans 2>/dev/null || true
        fi
    done

    echo -e "${GREEN}  Compose cleanup completed.${NC}"
}

validate_environment() {
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${RED}❌ Environment file not found: $ENV_FILE${NC}"
        echo "Please ensure the environment file exists before running bootstrap."
        exit 1
    fi
    echo -e "${GREEN}✅ Environment file found: $ENV_FILE${NC}"
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

build_compose_command() {
    local compose_cmd="docker compose --env-file $ENV_FILE"

    # Add infrastructure files
    if [ "$DEPLOY_INFRASTRUCTURE" = "true" ]; then
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.yml"
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.storage.yml"
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.monitoring.yml"
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.s3.yml"
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.etcd.yml"
    fi

    # Add gateway file
    if [ "$DEPLOY_GATEWAY" = "true" ] && [ -f "./Dockerfiles/docker-compose.gateway.yml" ]; then
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.gateway.yml"
    fi

    # Add extract executor file
    if [ "$DEPLOY_EXTRACT" = "true" ] && [ -f "./Dockerfiles/docker-compose.extract.yml" ]; then
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.extract.yml"
    fi

    # Add LiteLLM file
    if [ "$DEPLOY_LITELLM" = "true" ] && [ -f "./Dockerfiles/docker-compose.litellm.yml" ]; then
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
    fi

    compose_cmd="$compose_cmd --project-directory ."
    echo "$compose_cmd"
}

bootstrap_system() {
    echo ""
    echo -e "${BLUE}Starting Marie-AI system bootstrap...${NC}"

    # Source environment file
    source "$ENV_FILE"
    echo -e "${GREEN}✅ Environment loaded from $ENV_FILE${NC}"

    # Create network
    echo "Creating marie_default network..."
    docker network create marie_default 2>/dev/null || echo "Network marie_default already exists"

    # Create LiteLLM config directory if it doesn't exist
    if [ "$DEPLOY_LITELLM" = "true" ]; then
        mkdir -p "./Dockerfiles/litellm"
        if [ ! -f "./Dockerfiles/litellm/config.yaml" ]; then
            echo -e "${YELLOW}⚠️  LiteLLM config file not found. Creating default config...${NC}"
            echo "Please update ./Dockerfiles/litellm/config.yaml with your API keys and model configurations."
        fi
    fi

    # Build compose command
    local compose_cmd
    compose_cmd=$(build_compose_command)

    # Start services
    echo "Starting services with command:"
    echo "  $compose_cmd up --build --remove-orphans -d"
    echo ""

    eval "$compose_cmd up --build --remove-orphans -d"

    echo ""
    echo -e "${GREEN}🎉 Marie-AI system started successfully!${NC}"
    echo ""
    echo "Services status:"
    eval "$compose_cmd ps"

    # Show service endpoints
    show_service_endpoints
}

show_service_endpoints() {
    echo ""
    echo -e "${BLUE}🔗 Service Endpoints:${NC}"

    if [ "$DEPLOY_INFRASTRUCTURE" = "true" ]; then
        echo -e "${GREEN}Infrastructure Services:${NC}"
        echo "  🐰 RabbitMQ Management: http://localhost:15672 (guest/guest)"
        echo "  💾 MinIO Console: http://localhost:8001 (marieadmin/marietopsecret)"
        echo "  📊 Monitoring: http://localhost:3000"
    fi

    if [ "$DEPLOY_GATEWAY" = "true" ]; then
        echo -e "${GREEN}Gateway Services:${NC}"
        echo "  🌐 HTTP Gateway: http://localhost:52000"
        echo "  🔌 GRPC Gateway: grpc://localhost:51000"
    fi

    if [ "$DEPLOY_EXTRACT" = "true" ]; then
        echo -e "${GREEN}Extract Services:${NC}"
        echo "  🤖 Extract Executor: http://localhost:8080"
    fi

    if [ "$DEPLOY_LITELLM" = "true" ]; then
        echo -e "${GREEN}LiteLLM Services:${NC}"
        echo "  🤖 LiteLLM Proxy: http://localhost:4000"
        echo "  📊 LiteLLM Admin UI: http://localhost:4000/ui"
        echo "  🔧 LiteLLM Health: http://localhost:4000/health"
    fi
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
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
                shift
                ;;
            --no-litellm)
                DEPLOY_LITELLM=false
                shift
                ;;
            --infrastructure-only)
                DEPLOY_GATEWAY=false
                DEPLOY_EXTRACT=false
                DEPLOY_LITELLM=false
                shift
                ;;
            --services-only)
                DEPLOY_INFRASTRUCTURE=false
                shift
                ;;
            --litellm-only)
                DEPLOY_GATEWAY=false
                DEPLOY_EXTRACT=false
                DEPLOY_INFRASTRUCTURE=false
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
    echo "  --no-gateway          Skip gateway deployment"
    echo "  --no-extract          Skip extract executor deployment"
    echo "  --no-infrastructure   Skip infrastructure services"
    echo "  --no-litellm          Skip LiteLLM proxy deployment"
    echo "  --infrastructure-only Deploy only infrastructure services"
    echo "  --services-only       Deploy only Marie services (gateway + extract + litellm)"
    echo "  --litellm-only        Deploy only LiteLLM proxy"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy everything"
    echo "  $0 --infrastructure-only  # Deploy only infrastructure"
    echo "  $0 --services-only        # Deploy only Marie services"
    echo "  $0 --no-extract           # Deploy infrastructure + gateway + litellm only"
    echo "  $0 --litellm-only         # Deploy only LiteLLM proxy"
}

# Main execution flow
main() {
    parse_args "$@"

    show_deployment_config

    validate_environment
    validate_compose_files

    if check_running_services || check_compose_services; then
        prompt_cleanup
    else
        echo -e "${GREEN}✅ No conflicting services found.${NC}"
    fi

    # Bootstrap the system
    bootstrap_system

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Bootstrap completed successfully!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "Useful commands:"
    echo "  View logs: docker compose logs -f [service_name]"
    echo "  Stop services: docker compose down"
    echo "  Stop and cleanup: docker compose down --volumes --remove-orphans"
    echo "  Scale extract executors: docker compose up -d --scale marie-extract-executor=3"
    if [ "$DEPLOY_LITELLM" = "true" ]; then
        echo "  View LiteLLM logs: docker compose logs -f litellm"
        echo "  LiteLLM health check: curl http://localhost:4000/health"
    fi
}

# Run main function
main "$@"