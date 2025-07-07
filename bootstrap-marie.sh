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
ENV_FILE="./config/.env.dev"

DEPLOY_GATEWAY=${DEPLOY_GATEWAY:-true}
DEPLOY_EXTRACT=${DEPLOY_EXTRACT:-true}
DEPLOY_INFRASTRUCTURE=${DEPLOY_INFRASTRUCTURE:-true}
DEPLOY_LITELLM=${DEPLOY_LITELLM:-true}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Marie-AI System Bootstrap${NC}"
echo -e "${BLUE}========================================${NC}"

stop_all_services() {
    echo -e "${YELLOW}Stopping all Marie-AI services...${NC}"
    echo ""

    # Stop infrastructure services
    echo -e "${BLUE}🔧 Stopping infrastructure services...${NC}"
    docker compose --env-file $ENV_FILE \
        --project-name marie-infrastructure \
        -f ./Dockerfiles/docker-compose.storage.yml \
        -f ./Dockerfiles/docker-compose.s3.yml \
        -f ./Dockerfiles/docker-compose.rabbitmq.yml \
        -f ./Dockerfiles/docker-compose.etcd.yml \
        -f ./Dockerfiles/docker-compose.litellm.yml \
        --project-directory . \
        down --volumes --remove-orphans 2>/dev/null || echo "No infrastructure services to stop"

    # Stop application services
    echo -e "${BLUE}🚀 Stopping application services...${NC}"
    docker compose --env-file $ENV_FILE \
        --project-name marie-application \
        -f ./Dockerfiles/docker-compose.gateway.yml \
        -f ./Dockerfiles/docker-compose.extract.yml \
        --project-directory . \
        down --volumes --remove-orphans 2>/dev/null || echo "No application services to stop"

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

    # Clean up networks
    echo -e "${BLUE}🌐 Cleaning up networks...${NC}"
    if docker network ls | grep -q marie_default; then
        local network_containers
        network_containers=$(docker network inspect marie_default --format='{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || true)
        if [ -z "$network_containers" ]; then
            echo "Removing marie_default network..."
            docker network rm marie_default 2>/dev/null || true
        else
            echo "marie_default network still has containers, skipping removal"
        fi
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
    echo -e "    └── LLM Proxy (LiteLLM): ${DEPLOY_LITELLM}"
    echo -e "  Application Services:"
    echo -e "    ├── Gateway: ${DEPLOY_GATEWAY}"
    echo -e "    └── Extract Executors: ${DEPLOY_EXTRACT}"
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
    echo -e "${RED} Warning: Running services detected!${NC}"
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
        echo -e "${RED} Environment file not found: $ENV_FILE${NC}"
        echo "Please ensure the environment file exists before running bootstrap."
        exit 1
    fi
    echo -e "${GREEN} Environment file found: $ENV_FILE${NC}"
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
                echo -e "${YELLOW} Optional file missing: $compose_file${NC}"
            fi
        fi
    done

    if [ ${#missing_files[@]} -gt 0 ]; then
        echo -e "${RED} Missing required compose files:${NC}"
        for file in "${missing_files[@]}"; do
            echo "  - $file"
        done
        exit 1
    fi

    echo -e "${GREEN} All required compose files found.${NC}"
}

build_compose_command() {
    local compose_cmd="docker compose --env-file $ENV_FILE"

    if [ "$DEPLOY_INFRASTRUCTURE" = "true" ]; then
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.storage.yml"
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.s3.yml"
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.etcd.yml"

        # LiteLLM is part of infrastructure
        if [ "$DEPLOY_LITELLM" = "true" ] && [ -f "./Dockerfiles/docker-compose.litellm.yml" ]; then
            compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
        fi
    fi

    if [ "$DEPLOY_GATEWAY" = "true" ] && [ -f "./Dockerfiles/docker-compose.gateway.yml" ]; then
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.gateway.yml"
    fi

    if [ "$DEPLOY_EXTRACT" = "true" ] && [ -f "./Dockerfiles/docker-compose.extract.yml" ]; then
        compose_cmd="$compose_cmd -f ./Dockerfiles/docker-compose.extract.yml"
    fi

    compose_cmd="$compose_cmd --project-directory ."
    echo "$compose_cmd"
}

bootstrap_system() {
    echo ""
    echo -e "${BLUE}Starting Marie-AI system bootstrap...${NC}"

    source "$ENV_FILE"
    echo -e "${GREEN}✅ Environment loaded from $ENV_FILE${NC}"

    echo "Creating marie_default network..."
    docker network create marie_default 2>/dev/null || echo "Network marie_default already exists"

    # Stage 1: Start infrastructure services with separate project name
    if [ "$DEPLOY_INFRASTRUCTURE" = "true" ]; then
        echo -e "${BLUE}🔧 Stage 1: Starting infrastructure services...${NC}"

        local infra_compose_cmd="docker compose --env-file $ENV_FILE"
        infra_compose_cmd="$infra_compose_cmd --project-name marie-infrastructure"
        infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.storage.yml"
        infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.s3.yml"
        infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
        infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.etcd.yml"

        if [ "$DEPLOY_LITELLM" = "true" ] && [ -f "./Dockerfiles/docker-compose.litellm.yml" ]; then
            infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
        fi

        infra_compose_cmd="$infra_compose_cmd --project-directory ."

        echo "Starting infrastructure services..."
        eval "$infra_compose_cmd up -d --build --remove-orphans"

        echo -e "${YELLOW}⏳ Waiting for infrastructure services to be healthy...${NC}"
        eval "$infra_compose_cmd up --wait"

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
        echo "Starting application services..."
        # Now --remove-orphans won't affect infrastructure services
        eval "$app_compose_cmd up -d --build --remove-orphans"
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
        docker compose --env-file $ENV_FILE \
            --project-name marie-infrastructure \
            -f ./Dockerfiles/docker-compose.storage.yml \
            -f ./Dockerfiles/docker-compose.s3.yml \
            -f ./Dockerfiles/docker-compose.rabbitmq.yml \
            -f ./Dockerfiles/docker-compose.etcd.yml \
            -f ./Dockerfiles/docker-compose.litellm.yml \
            --project-directory . \
            ps 2>/dev/null || echo "No infrastructure services"
    fi

    if [ "$DEPLOY_GATEWAY" = "true" ] || [ "$DEPLOY_EXTRACT" = "true" ]; then
        echo ""
        echo -e "${BLUE}Application Services:${NC}"
        docker compose --env-file $ENV_FILE \
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
        echo "  🐰 RabbitMQ Management: http://localhost:15672 (guest/guest)"
        echo "  💾 MinIO Console: http://localhost:8001 (marieadmin/marietopsecret)"
        echo "  📊 Monitoring: http://localhost:3000"
        echo "  🗄️  etcd: http://localhost:2379"

        if [ "$DEPLOY_LITELLM" = "true" ]; then
            echo "  🤖 LiteLLM Proxy: http://localhost:4000"
            echo "  📊 LiteLLM Admin UI: http://localhost:4000/ui"
            echo "  🔧 LiteLLM Health: http://localhost:4000/health"
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
    echo "  --stop-all            Stop and remove all Marie-AI services and containers"
    echo "  --no-gateway          Skip gateway deployment"
    echo "  --no-extract          Skip extract executor deployment"
    echo "  --no-infrastructure   Skip infrastructure services (includes LiteLLM)"
    echo "  --no-litellm          Skip LiteLLM proxy deployment"
    echo "  --infrastructure-only Deploy only infrastructure services (includes LiteLLM)"
    echo "  --services-only       Deploy only Marie application services (gateway + extract)"
    echo "  --litellm-only        Deploy only LiteLLM proxy (with required infrastructure)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Service Categories:"
    echo "  Infrastructure: Storage, Message Queue, Service Discovery, LLM Proxy"
    echo "  Application:    Gateway, Extract Executors"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy everything"
    echo "  $0 --stop-all         # Stop all services and cleanup"
    echo "  $0 --infrastructure-only  # Deploy infrastructure + LiteLLM"
    echo "  $0 --services-only        # Deploy only gateway + extract"
    echo "  $0 --no-extract           # Deploy infrastructure + gateway only"
    echo "  $0 --litellm-only         # Deploy minimal infrastructure + LiteLLM"
}

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
    echo "  Scale extract executors: docker compose up -d --scale marie-extract-executor=3"
    if [ "$DEPLOY_LITELLM" = "true" ]; then
        echo "  View LiteLLM logs: docker compose logs -f litellm"
        echo "  LiteLLM health check: curl http://localhost:4000/health"
    fi
}

main "$@"