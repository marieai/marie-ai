
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
            docker ps --format "table {{.Names}}\t{{.Status}}" --filter "name=marie-s3-server" --filter "name=marie-psql-server" --filter "name=marie-rabbitmq" --filter "name=etcd-single" --filter "name=marie-litellm" --filter "name=marie-mc-setup" 2>/dev/null | tail -n +2
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
        --project-directory . \
        down --volumes $orphan_flag 2>/dev/null || echo "No infrastructure services to stop"

    # Stop any remaining infrastructure containers
    local containers
    containers=$(docker ps -q --filter "name=marie-s3-server" --filter "name=marie-psql-server" --filter "name=marie-rabbitmq" --filter "name=etcd-single" --filter "name=marie-litellm" --filter "name=marie-mc-setup" 2>/dev/null || true)
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
    echo -e "    └── LLM Proxy (LiteLLM): ${DEPLOY_LITELLM}"
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

        if [ "$DEPLOY_LITELLM" = "true" ] && [ -f "./Dockerfiles/docker-compose.litellm.yml" ]; then
            infra_compose_cmd="$infra_compose_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
        fi

        infra_compose_cmd="$infra_compose_cmd --project-directory ."

        echo "Starting infrastructure services with host networking..."
        # Use host networking for all services
        COMPOSE_NETWORK_MODE=host eval "$infra_compose_cmd up -d --build $orphan_flag"

        echo -e "${YELLOW}⏳ Waiting for infrastructure services to be healthy (excluding setup containers)...${NC}"

        # Build list of services to wait for (excluding setup containers)
        local services_to_wait=("s3server" "psql" "rabbitmq" "etcd-single")

        if [ "$DEPLOY_LITELLM" = "true" ]; then
            services_to_wait+=("litellm")
        fi

        # Wait only for specific services, excluding setup containers
        COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE \
            --project-name marie-infrastructure \
            -f ./Dockerfiles/docker-compose.storage.yml \
            -f ./Dockerfiles/docker-compose.s3.yml \
            -f ./Dockerfiles/docker-compose.rabbitmq.yml \
            -f ./Dockerfiles/docker-compose.etcd.yml \
            $([ "$DEPLOY_LITELLM" = "true" ] && echo "-f ./Dockerfiles/docker-compose.litellm.yml") \
            --project-directory . \
            up --wait ${services_to_wait[@]}

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
        COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE \
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
        echo "  🐰 RabbitMQ Management: http://localhost:15672 (guest/guest)"
        echo "  💾 MinIO S3 API: http://localhost:9000 (marieadmin/marietopsecret)"
        echo "  💾 MinIO Console: http://localhost:9001 (marieadmin/marietopsecret)"
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