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
DEPLOY_CLICKSTACK=${DEPLOY_CLICKSTACK:-false}
DEPLOY_GITEA=${DEPLOY_GITEA:-true}
# Note: Mem0 is now integrated as a Python SDK, not a separate container
# The database is always created in initialize_databases()

# Vagrant configuration
VAGRANT_MODE=${VAGRANT_MODE:-false}
VAGRANT_DIR="./vagrant"
VAGRANT_INSTANCE="${VAGRANT_INSTANCE:-1}"
VAGRANT_VM_NAME="${VAGRANT_VM_NAME:-marie-test-${VAGRANT_INSTANCE}}"
VAGRANT_SYNC_IMAGES=${VAGRANT_SYNC_IMAGES:-false}

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
            docker ps --format "table {{.Names}}\t{{.Status}}" --filter "name=marie-s3-server" --filter "name=marie-psql-server" --filter "name=marie-rabbitmq" --filter "name=etcd-single" --filter "name=marie-litellm" --filter "name=marie-mc-setup" --filter "name=marie-clickhouse" --filter "name=marie-hyperdx" --filter "name=marie-log-collector" --filter "name=marie-gitea" 2>/dev/null | tail -n +2
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

    # Stop infrastructure services (includes ClickStack if compose file exists)
    echo -e "${BLUE}🔧 Stopping infrastructure services...${NC}"
    local stop_cmd="COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE"
    stop_cmd="$stop_cmd --project-name marie-infrastructure"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.storage.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.s3.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.etcd.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.clickhouse.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.gitea.yml"
    if [ -f "./Dockerfiles/docker-compose.clickstack.yml" ]; then
        stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.clickstack.yml"
    fi
    stop_cmd="$stop_cmd --project-directory ."
    eval "$stop_cmd down --volumes $orphan_flag" 2>/dev/null || echo "No infrastructure services to stop"

    # Stop any remaining infrastructure containers
    local containers
    containers=$(docker ps -q --filter "name=marie-s3-server" --filter "name=marie-psql-server" --filter "name=marie-rabbitmq" --filter "name=etcd-single" --filter "name=marie-litellm" --filter "name=marie-mc-setup" --filter "name=marie-clickhouse" --filter "name=marie-hyperdx" --filter "name=marie-log-collector" --filter "name=marie-gitea" 2>/dev/null || true)
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

    local stop_cmd="COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE"
    stop_cmd="$stop_cmd --project-name marie-infrastructure"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.storage.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.s3.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.etcd.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.clickhouse.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.gitea.yml"
    if [ -f "./Dockerfiles/docker-compose.clickstack.yml" ]; then
        stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.clickstack.yml"
    fi
    stop_cmd="$stop_cmd --project-directory ."
    eval "$stop_cmd down $orphan_flag" 2>/dev/null || echo "No infrastructure services to stop"

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

    # Stop infrastructure services (includes ClickStack)
    echo -e "${BLUE}🔧 Stopping infrastructure services...${NC}"
    local stop_cmd="COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE"
    stop_cmd="$stop_cmd --project-name marie-infrastructure"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.storage.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.s3.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.etcd.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.clickhouse.yml"
    stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.gitea.yml"
    if [ -f "./Dockerfiles/docker-compose.clickstack.yml" ]; then
        stop_cmd="$stop_cmd -f ./Dockerfiles/docker-compose.clickstack.yml"
    fi
    stop_cmd="$stop_cmd --project-directory ."
    eval "$stop_cmd down --volumes $orphan_flag" 2>/dev/null || echo "No infrastructure services to stop"

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

# ============================================================
# Vagrant Functions
# ============================================================

vagrant_check_installed() {
    if ! command -v vagrant &> /dev/null; then
        echo -e "${RED}❌ Vagrant is not installed${NC}"
        echo "Please install Vagrant: https://www.vagrantup.com/downloads"
        echo ""
        echo "Quick install:"
        echo "  macOS: brew install --cask vagrant"
        echo "  Ubuntu: sudo apt-get install vagrant"
        echo "  Fedora: sudo dnf install vagrant"
        exit 1
    fi

    if ! command -v VBoxManage &> /dev/null && ! command -v virsh &> /dev/null; then
        echo -e "${YELLOW}⚠️  No supported VM provider found${NC}"
        echo "Please install VirtualBox or libvirt:"
        echo "  VirtualBox: https://www.virtualbox.org/wiki/Downloads"
        echo "  libvirt: sudo apt-get install libvirt-daemon-system"
    fi
}

vagrant_check_libvirt_deps() {
    # Check for libvirt development libraries needed for vagrant-libvirt plugin
    local missing_deps=()

    # Check for libvirt-dev (needed for plugin compilation)
    if ! pkg-config --exists libvirt 2>/dev/null; then
        missing_deps+=("libvirt-dev")
    fi

    # Check for libvirt daemon
    if ! command -v libvirtd &> /dev/null && ! systemctl list-unit-files libvirtd.service &> /dev/null; then
        missing_deps+=("libvirt-daemon-system")
    fi

    # Check for virsh client
    if ! command -v virsh &> /dev/null; then
        missing_deps+=("libvirt-clients")
    fi

    # Check for qemu-kvm
    if ! command -v qemu-system-x86_64 &> /dev/null && ! command -v kvm &> /dev/null; then
        missing_deps+=("qemu-kvm")
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        echo -e "${YELLOW}========================================${NC}" >&2
        echo -e "${YELLOW}  Missing libvirt Dependencies${NC}" >&2
        echo -e "${YELLOW}========================================${NC}" >&2
        echo "" >&2
        echo "The following packages are required for vagrant-libvirt:" >&2
        echo "" >&2

        # Detect package manager and provide appropriate command
        if command -v apt-get &> /dev/null; then
            echo "  sudo apt-get install -y ${missing_deps[*]}" >&2
        elif command -v dnf &> /dev/null; then
            # Map to Fedora package names
            local fedora_deps=()
            for dep in "${missing_deps[@]}"; do
                case "$dep" in
                    libvirt-dev) fedora_deps+=("libvirt-devel") ;;
                    libvirt-daemon-system) fedora_deps+=("libvirt-daemon") ;;
                    libvirt-clients) fedora_deps+=("libvirt-client") ;;
                    *) fedora_deps+=("$dep") ;;
                esac
            done
            echo "  sudo dnf install -y ${fedora_deps[*]}" >&2
        elif command -v pacman &> /dev/null; then
            echo "  sudo pacman -S libvirt qemu virt-manager" >&2
        else
            echo "  Install: ${missing_deps[*]}" >&2
        fi
        echo "" >&2
        return 1
    fi
    return 0
}

vagrant_detect_provider() {
    # Check if provider is already set
    if [ -n "$VAGRANT_DEFAULT_PROVIDER" ]; then
        echo "$VAGRANT_DEFAULT_PROVIDER"
        return
    fi

    # Check if KVM is loaded (Linux only)
    if [ -f /proc/modules ] && grep -q "^kvm" /proc/modules 2>/dev/null; then
        # KVM is loaded - VirtualBox won't work
        echo -e "${BLUE}KVM detected, using libvirt provider${NC}" >&2

        # Check libvirt dependencies
        if ! vagrant_check_libvirt_deps; then
            echo "" >&2
            echo "After installing dependencies, also run:" >&2
            echo "  sudo usermod -aG libvirt \$USER" >&2
            echo "  newgrp libvirt  # or log out and back in" >&2
            echo "  vagrant plugin install vagrant-libvirt" >&2
            echo "" >&2
            exit 1
        fi

        # Check if user is in libvirt group (check /etc/group, not current session)
        local current_user
        current_user=$(whoami)
        local in_libvirt_group=false

        # Check if user is in libvirt or libvirt-qemu group (varies by distro)
        if getent group libvirt 2>/dev/null | grep -q "\b${current_user}\b"; then
            in_libvirt_group=true
        elif getent group libvirt-qemu 2>/dev/null | grep -q "\b${current_user}\b"; then
            in_libvirt_group=true
        elif getent group kvm 2>/dev/null | grep -q "\b${current_user}\b"; then
            in_libvirt_group=true
        fi

        if [ "$in_libvirt_group" = false ]; then
            echo -e "${YELLOW}========================================${NC}" >&2
            echo -e "${YELLOW}  User Not in libvirt Group${NC}" >&2
            echo -e "${YELLOW}========================================${NC}" >&2
            echo "" >&2
            echo "Add yourself to the libvirt group:" >&2
            echo "  sudo usermod -aG libvirt \$USER" >&2
            echo "" >&2
            echo "Then either log out and back in, or run:" >&2
            echo "  newgrp libvirt" >&2
            echo "" >&2
            exit 1
        fi

        # Check if current session has the group (might need newgrp or re-login)
        if ! groups | grep -qE "(libvirt|libvirt-qemu|kvm)"; then
            echo -e "${YELLOW}========================================${NC}" >&2
            echo -e "${YELLOW}  libvirt Group Not Active in Session${NC}" >&2
            echo -e "${YELLOW}========================================${NC}" >&2
            echo "" >&2
            echo "You are in the libvirt group but it's not active in this session." >&2
            echo "Either log out and back in, or run:" >&2
            echo "  newgrp libvirt" >&2
            echo "  ./bootstrap-marie.sh --vagrant" >&2
            echo "" >&2
            exit 1
        fi

        # Check if vagrant-libvirt plugin is installed
        if ! vagrant plugin list 2>/dev/null | grep -q "vagrant-libvirt"; then
            echo -e "${YELLOW}========================================${NC}" >&2
            echo -e "${YELLOW}  vagrant-libvirt Plugin Required${NC}" >&2
            echo -e "${YELLOW}========================================${NC}" >&2
            echo "" >&2
            echo "Install the vagrant-libvirt plugin:" >&2
            echo "  vagrant plugin install vagrant-libvirt" >&2
            echo "" >&2
            exit 1
        fi

        echo "libvirt"
        return
    fi

    # Default to virtualbox if available
    if command -v VBoxManage &> /dev/null; then
        echo "virtualbox"
        return
    fi

    # Fall back to libvirt if available
    if command -v virsh &> /dev/null; then
        if ! vagrant_check_libvirt_deps; then
            exit 1
        fi
        if vagrant plugin list 2>/dev/null | grep -q "vagrant-libvirt"; then
            echo "libvirt"
            return
        fi
    fi

    # No provider found
    echo -e "${RED}========================================${NC}" >&2
    echo -e "${RED}  No Vagrant Provider Available${NC}" >&2
    echo -e "${RED}========================================${NC}" >&2
    echo "" >&2
    echo "Install VirtualBox or libvirt:" >&2
    echo "" >&2
    echo "Option 1 - VirtualBox:" >&2
    echo "  https://www.virtualbox.org/wiki/Downloads" >&2
    echo "" >&2
    echo "Option 2 - libvirt (recommended for Linux with KVM):" >&2
    if command -v apt-get &> /dev/null; then
        echo "  sudo apt-get install -y libvirt-dev libvirt-daemon-system libvirt-clients qemu-kvm" >&2
    elif command -v dnf &> /dev/null; then
        echo "  sudo dnf install -y libvirt-devel libvirt-daemon libvirt-client qemu-kvm" >&2
    else
        echo "  Install: libvirt-dev libvirt-daemon-system libvirt-clients qemu-kvm" >&2
    fi
    echo "  sudo usermod -aG libvirt \$USER" >&2
    echo "  newgrp libvirt" >&2
    echo "  vagrant plugin install vagrant-libvirt" >&2
    echo "" >&2
    exit 1
}

vagrant_filter_output() {
    # Filter vagrant output to show only important messages
    while IFS= read -r line; do
        # Skip verbose/noisy lines
        case "$line" in
            *"Vagrant insecure key"*|*"Inserting generated public"*|*"Removing insecure key"*|*"Key inserted"*)
                # Skip SSH key messages
                ;;
            *"Warning: Connection refused"*)
                # Show connection status on same line
                printf "\r  ⏳ Waiting for SSH...        "
                ;;
            *"SSH address:"*)
                local ip=$(echo "$line" | grep -oP '\d+\.\d+\.\d+\.\d+')
                printf "\r  📍 VM IP: $ip             \n"
                ;;
            *"Waiting for domain"*|*"Waiting for machine"*)
                printf "  ⏳ Booting VM...\n"
                ;;
            *"Machine booted and ready"*)
                printf "  ✅ VM booted\n"
                ;;
            *"Setting hostname"*)
                printf "  ✅ Hostname configured\n"
                ;;
            *"Forwarding ports"*)
                printf "  ✅ Port forwarding:\n"
                ;;
            *"(guest) =>"*)
                # Port forwarding detail - show condensed
                local guest=$(echo "$line" | grep -oP '\d+(?= \(guest\))')
                local host=$(echo "$line" | grep -oP '(?<=> )\d+(?= \(host\))')
                if [ -n "$guest" ] && [ -n "$host" ]; then
                    printf "      $guest → $host\n"
                fi
                ;;
            *"Rsyncing folder"*)
                local folder=$(echo "$line" | grep -oP '(?<=> )/[^ ]+' | head -1)
                printf "  ✅ Syncing: ${folder}\n"
                ;;
            *"Creating domain"*)
                printf "  ⏳ Creating VM...\n"
                ;;
            *"Creating image"*)
                printf "  ⏳ Preparing disk image...\n"
                ;;
            *"Starting domain"*)
                printf "  ⏳ Starting VM...\n"
                ;;
            *"Downloading:"*)
                printf "  ⏳ Downloading box (this may take a while)...\n"
                ;;
            *"Successfully added box"*)
                printf "  ✅ Box downloaded\n"
                ;;
            *"Uploading base box"*)
                printf "  ⏳ Uploading box to libvirt storage...\n"
                ;;
            *"Error"*|*"error:"*)
                # Always show errors
                echo -e "${RED}  ❌ $line${NC}"
                ;;
            *"[fog][WARNING]"*)
                # Skip fog warnings
                ;;
            "")
                # Skip empty lines
                ;;
            *"-- Name:"*)
                local vmname=$(echo "$line" | awk '{print $NF}')
                printf "  📦 VM: $vmname\n"
                ;;
            *"-- Cpus:"*|*"-- Memory:"*)
                # Show resources
                local val=$(echo "$line" | awk '{print $NF}')
                local key=$(echo "$line" | grep -oP '(?<=-- )[^:]+')
                printf "      $key: $val\n"
                ;;
            *)
                # Skip most other lines
                :
                ;;
        esac
    done
}

vagrant_status() {
    vagrant_check_installed

    if [ ! -d "$VAGRANT_DIR" ]; then
        echo -e "${RED}❌ Vagrant directory not found: $VAGRANT_DIR${NC}"
        exit 1
    fi

    # Detect provider (but don't fail if none available for status check)
    local provider
    provider=$(vagrant_detect_provider 2>/dev/null || echo "unknown")

    # Calculate port offset for this instance
    local port_offset=$((10000 + (VAGRANT_INSTANCE - 1) * 1000))
    local vm_ip="192.168.56.$((10 + VAGRANT_INSTANCE))"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    Vagrant VM Status (Instance $VAGRANT_INSTANCE)${NC}"
    echo -e "${BLUE}    Provider: $provider${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    cd "$VAGRANT_DIR"
    VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant status
    echo ""

    # Check if VM is running and show additional info
    local vm_status
    vm_status=$(VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant status --machine-readable 2>/dev/null | grep ",state," | cut -d',' -f4)

    if [ "$vm_status" = "running" ]; then
        echo -e "${GREEN}VM Instance $VAGRANT_INSTANCE is running${NC}"
        echo ""
        echo "SSH into VM:    ./bootstrap-marie.sh --vagrant-ssh --instance=$VAGRANT_INSTANCE"
        echo ""
        echo "Port mappings (Host -> VM) for Instance $VAGRANT_INSTANCE:"
        echo "  PostgreSQL:     localhost:$((5432 + port_offset))"
        echo "  ClickHouse:     localhost:$((8123 + port_offset))"
        echo "  RabbitMQ AMQP:  localhost:$((5673 + port_offset))"
        echo "  RabbitMQ Mgmt:  localhost:$((15672 + port_offset))"
        echo "  MinIO Console:  localhost:$((9002 + port_offset))"
        echo "  Gitea:          localhost:$((3001 + port_offset))"
        echo "  LiteLLM:        localhost:$((4000 + port_offset))"
        echo "  Gateway HTTP:   localhost:$((52000 + port_offset))"
        echo ""
        echo "Direct VM access: http://$vm_ip"
    fi

    cd - > /dev/null
}

vagrant_up() {
    vagrant_check_installed

    if [ ! -d "$VAGRANT_DIR" ]; then
        echo -e "${RED}❌ Vagrant directory not found: $VAGRANT_DIR${NC}"
        exit 1
    fi

    # Detect and set provider
    local provider
    provider=$(vagrant_detect_provider)
    export VAGRANT_DEFAULT_PROVIDER="$provider"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    Starting Vagrant VM (Instance $VAGRANT_INSTANCE)${NC}"
    echo -e "${BLUE}    Provider: $provider${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    cd "$VAGRANT_DIR"

    # Check current status
    local vm_status
    vm_status=$(VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant status --machine-readable 2>/dev/null | grep ",state," | cut -d',' -f4)

    if [ "$vm_status" = "running" ]; then
        echo -e "${GREEN}✅ VM Instance $VAGRANT_INSTANCE is already running${NC}"
        echo "Syncing files..."
        VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant rsync 2>&1 | vagrant_filter_output
    else
        echo "Starting VM Instance $VAGRANT_INSTANCE (this may take a few minutes on first run)..."
        VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant up --provider="$provider" 2>&1 | vagrant_filter_output
    fi

    cd - > /dev/null

    echo ""
    echo -e "${GREEN}✅ Vagrant VM Instance $VAGRANT_INSTANCE is ready${NC}"
}

vagrant_down() {
    vagrant_check_installed

    if [ ! -d "$VAGRANT_DIR" ]; then
        echo -e "${RED}❌ Vagrant directory not found: $VAGRANT_DIR${NC}"
        exit 1
    fi

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}    Destroying Vagrant VM (Instance $VAGRANT_INSTANCE)${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  This will permanently delete VM Instance $VAGRANT_INSTANCE and all data inside it.${NC}"
    read -p "Are you sure? (y/N): " confirm

    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Cancelled."
        exit 0
    fi

    cd "$VAGRANT_DIR"
    VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant destroy -f
    cd - > /dev/null

    echo ""
    echo -e "${GREEN}✅ Vagrant VM Instance $VAGRANT_INSTANCE destroyed${NC}"
}

vagrant_ssh() {
    vagrant_check_installed

    if [ ! -d "$VAGRANT_DIR" ]; then
        echo -e "${RED}❌ Vagrant directory not found: $VAGRANT_DIR${NC}"
        exit 1
    fi

    # Detect and set provider
    local provider
    provider=$(vagrant_detect_provider)
    export VAGRANT_DEFAULT_PROVIDER="$provider"

    cd "$VAGRANT_DIR"

    # Check if VM is running
    local vm_status
    vm_status=$(VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant status --machine-readable 2>/dev/null | grep ",state," | cut -d',' -f4)

    if [ "$vm_status" != "running" ]; then
        echo -e "${YELLOW}VM Instance $VAGRANT_INSTANCE is not running. Starting VM...${NC}"
        VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant up --provider="$provider" 2>&1 | vagrant_filter_output
    fi

    echo -e "${BLUE}Connecting to Vagrant VM Instance $VAGRANT_INSTANCE...${NC}"
    VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant ssh

    cd - > /dev/null
}

vagrant_ensure_running() {
    vagrant_check_installed

    if [ ! -d "$VAGRANT_DIR" ]; then
        echo -e "${RED}❌ Vagrant directory not found: $VAGRANT_DIR${NC}"
        exit 1
    fi

    # Detect and set provider
    local provider
    provider=$(vagrant_detect_provider)
    export VAGRANT_DEFAULT_PROVIDER="$provider"

    cd "$VAGRANT_DIR"

    local vm_status
    vm_status=$(VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant status --machine-readable 2>/dev/null | grep ",state," | cut -d',' -f4)

    if [ "$vm_status" != "running" ]; then
        echo -e "${YELLOW}VM Instance $VAGRANT_INSTANCE is not running. Starting VM...${NC}"
        echo -e "${BLUE}Provider: $provider${NC}"
        VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant up --provider="$provider" 2>&1 | vagrant_filter_output
    else
        # Sync files to ensure latest configs are in VM
        echo -e "${BLUE}Syncing files to VM Instance $VAGRANT_INSTANCE...${NC}"
        VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant rsync 2>&1 | vagrant_filter_output
    fi

    cd - > /dev/null
}

vagrant_exec() {
    local cmd="$1"
    cd "$VAGRANT_DIR"
    VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant ssh -c "$cmd"
    local exit_code=$?
    cd - > /dev/null
    return $exit_code
}

vagrant_list() {
    vagrant_check_installed

    if [ ! -d "$VAGRANT_DIR" ]; then
        echo -e "${RED}❌ Vagrant directory not found: $VAGRANT_DIR${NC}"
        exit 1
    fi

    # Detect provider
    local provider
    provider=$(vagrant_detect_provider 2>/dev/null || echo "unknown")

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    Marie-AI Vagrant Instances${NC}"
    echo -e "${BLUE}    Provider: $provider${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    cd "$VAGRANT_DIR"

    # Check status of all possible instances (1-9)
    local found_any=false
    echo -e "${BLUE}Instance  VM Name          IP Address       Status     PostgreSQL  ClickHouse${NC}"
    echo "--------  ---------------  ---------------  ---------  ----------  ----------"

    for i in 1 2 3 4 5 6 7 8 9; do
        local vm_status
        vm_status=$(VAGRANT_INSTANCE=$i vagrant status --machine-readable 2>/dev/null | grep ",state," | cut -d',' -f4)

        if [ -n "$vm_status" ] && [ "$vm_status" != "not_created" ]; then
            found_any=true
            local port_offset=$((10000 + (i - 1) * 1000))
            local vm_ip="192.168.56.$((10 + i))"
            local pg_port=$((5432 + port_offset))
            local ch_port=$((8123 + port_offset))

            # Color based on status
            local status_color="${YELLOW}"
            if [ "$vm_status" = "running" ]; then
                status_color="${GREEN}"
            elif [ "$vm_status" = "poweroff" ] || [ "$vm_status" = "saved" ]; then
                status_color="${RED}"
            fi

            printf "%-8s  %-15s  %-15s  ${status_color}%-9s${NC}  %-10s  %-10s\n" \
                "$i" "marie-test-$i" "$vm_ip" "$vm_status" "$pg_port" "$ch_port"
        fi
    done

    if [ "$found_any" = false ]; then
        echo -e "${YELLOW}No Vagrant instances found.${NC}"
        echo ""
        echo "Create a new instance:"
        echo "  ./bootstrap-marie.sh --vagrant                  # Instance 1 (default)"
        echo "  ./bootstrap-marie.sh --vagrant --instance=2     # Instance 2"
    else
        echo ""
        echo "Commands:"
        echo "  ./bootstrap-marie.sh --vagrant-status --instance=N   # Check instance N"
        echo "  ./bootstrap-marie.sh --vagrant-ssh --instance=N      # SSH into instance N"
        echo "  ./bootstrap-marie.sh --vagrant-down --instance=N     # Destroy instance N"
    fi

    cd - > /dev/null
}

vagrant_sync_images() {
    # Stream local Docker images directly to the Vagrant VM (no temp files on host)
    # This avoids disk space issues by piping: docker save | ssh docker load

    local images_to_sync=("$@")

    if [ ${#images_to_sync[@]} -eq 0 ]; then
        # Default: sync all infrastructure images that exist locally
        images_to_sync=(
            "ghcr.io/ferretdb/postgres-documentdb:17-0.103.0"
            "rabbitmq:3-management-alpine"
            "minio/minio:latest"
            "minio/mc:latest"
            "quay.io/coreos/etcd:v3.6.1"
            "gitea/gitea:latest"
            "clickhouse/clickhouse-server:latest"
            "marieai/marie-gateway:4.0.0-cpu"
            "marieai/marie:4.0.0-cuda"
            "marieai/marie:4.0.0-cpu"
        )
    fi

    echo -e "${BLUE}Streaming Docker images to VM...${NC}"

    cd "$VAGRANT_DIR"

    local synced=0
    local skipped=0
    local failed=0

    for img in "${images_to_sync[@]}"; do
        if docker image inspect "$img" &>/dev/null; then
            local img_size
            img_size=$(docker image inspect "$img" --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || echo "unknown")
            echo -e "${BLUE}⏳ Streaming $img ($img_size)...${NC}"

            # Stream via vagrant ssh (uses Vagrant's SSH config automatically)
            # Use subshell to prevent set -e from exiting on pipe failure
            local result=0
            docker save "$img" | VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant ssh -c "docker load" || result=$?
            if [ $result -eq 0 ]; then
                echo -e "${GREEN}✔ $img${NC}"
                ((synced++)) || true
            else
                echo -e "${RED}❌ Failed: $img${NC}"
                ((failed++)) || true
            fi
        else
            ((skipped++)) || true
        fi
    done

    cd - > /dev/null

    echo -e "${GREEN}✔ Synced $synced image(s)${NC}"
    if [ $skipped -gt 0 ]; then
        echo -e "${YELLOW}ℹ Skipped $skipped image(s) (not found locally)${NC}"
    fi
    if [ $failed -gt 0 ]; then
        echo -e "${YELLOW}⚠ Failed: $failed image(s)${NC}"
    fi
}

vagrant_bootstrap() {
    # Calculate port offset for this instance
    local port_offset=$((10000 + (VAGRANT_INSTANCE - 1) * 1000))
    local vm_ip="192.168.56.$((10 + VAGRANT_INSTANCE))"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    Marie-AI Vagrant Deployment${NC}"
    echo -e "${BLUE}    Instance: $VAGRANT_INSTANCE${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Ensure VM is running
    vagrant_ensure_running

    # Sync local Docker images to VM if requested
    if [ "$VAGRANT_SYNC_IMAGES" = "true" ]; then
        vagrant_sync_images
    fi

    # Build the command to run inside the VM
    local vagrant_args=""

    # Pass through relevant flags (excluding vagrant-specific ones)
    if [ "$DEPLOY_GATEWAY" = "false" ]; then
        vagrant_args="$vagrant_args --no-gateway"
    fi
    if [ "$DEPLOY_EXTRACT" = "false" ]; then
        vagrant_args="$vagrant_args --no-extract"
    fi
    if [ "$DEPLOY_INFRASTRUCTURE" = "false" ]; then
        vagrant_args="$vagrant_args --no-infrastructure"
    fi
    if [ "$DEPLOY_LITELLM" = "true" ]; then
        # LiteLLM is disabled by default, only pass if explicitly enabled
        : # Don't add --no-litellm
    else
        vagrant_args="$vagrant_args --no-litellm"
    fi
    if [ "$DEPLOY_CLICKHOUSE" = "false" ]; then
        vagrant_args="$vagrant_args --no-clickhouse"
    fi
    if [ "$DEPLOY_CLICKSTACK" = "true" ]; then
        vagrant_args="$vagrant_args --with-clickstack"
    fi
    if [ "$DEPLOY_GITEA" = "false" ]; then
        vagrant_args="$vagrant_args --no-gitea"
    fi
    # Note: Mem0 is SDK-based, no container deployment needed

    # Determine which env file to use
    local vm_env_file="/home/vagrant/marie/config/.env.dev"

    if [ -n "$ENV_FILE" ] && [ "$ENV_FILE" != "./config/.env.dev" ]; then
        # Custom env file specified - need to copy it to VM
        local env_basename
        env_basename=$(basename "$ENV_FILE")
        echo -e "${BLUE}Copying environment file to VM...${NC}"
        cd "$VAGRANT_DIR"
        VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant ssh -c "mkdir -p /home/vagrant/marie/config"
        # Use vagrant rsync to sync the config dir which should include the env file
        VAGRANT_INSTANCE=$VAGRANT_INSTANCE vagrant rsync
        vm_env_file="/home/vagrant/marie/config/$env_basename"
        cd - > /dev/null
    fi

    echo -e "${BLUE}Running bootstrap inside Vagrant VM Instance $VAGRANT_INSTANCE...${NC}"
    echo ""

    # Execute bootstrap inside VM
    vagrant_exec "cd /home/vagrant/marie && ./bootstrap-marie.sh --env-file $vm_env_file $vagrant_args"

    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}    Vagrant Deployment Complete!${NC}"
        echo -e "${GREEN}    Instance: $VAGRANT_INSTANCE${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "Services are running inside Vagrant VM Instance $VAGRANT_INSTANCE."
        echo ""
        echo "Access services via forwarded ports:"
        echo "  PostgreSQL:     localhost:$((5432 + port_offset))"
        echo "  ClickHouse:     localhost:$((8123 + port_offset))"
        echo "  RabbitMQ Mgmt:  http://localhost:$((15672 + port_offset))"
        echo "  MinIO Console:  http://localhost:$((9002 + port_offset))"
        echo "  Gitea:          http://localhost:$((3001 + port_offset))"
        echo "  LiteLLM:        http://localhost:$((4000 + port_offset))"
        echo "  Gateway HTTP:   http://localhost:$((52000 + port_offset))"
        echo ""
        echo "Direct VM access: http://$vm_ip"
        echo ""
        echo "SSH into VM:      ./bootstrap-marie.sh --vagrant-ssh --instance=$VAGRANT_INSTANCE"
        echo "Check status:     ./bootstrap-marie.sh --vagrant-status --instance=$VAGRANT_INSTANCE"
        echo "Destroy VM:       ./bootstrap-marie.sh --vagrant-down --instance=$VAGRANT_INSTANCE"
        echo "List all VMs:     ./bootstrap-marie.sh --vagrant-list"
    else
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}    Vagrant Deployment Failed${NC}"
        echo -e "${RED}    Instance: $VAGRANT_INSTANCE${NC}"
        echo -e "${RED}========================================${NC}"
        echo ""
        echo "SSH into VM to investigate:"
        echo "  ./bootstrap-marie.sh --vagrant-ssh --instance=$VAGRANT_INSTANCE"
    fi

    return $exit_code
}

show_deployment_config() {
    echo -e "${BLUE}Deployment Configuration:${NC}"
    echo -e "  Infrastructure: ${DEPLOY_INFRASTRUCTURE}"
    echo -e "    ├── Storage (MinIO): ${DEPLOY_INFRASTRUCTURE}"
    echo -e "    ├── Message Queue (RabbitMQ): ${DEPLOY_INFRASTRUCTURE}"
    echo -e "    ├── Service Discovery (etcd): ${DEPLOY_INFRASTRUCTURE}"
    echo -e "    ├── LLM Proxy (LiteLLM): ${DEPLOY_LITELLM}"
    echo -e "    ├── Analytics DB (ClickHouse): ${DEPLOY_CLICKHOUSE}"
    echo -e "    ├── Observability (ClickStack): ${DEPLOY_CLICKSTACK}"
    echo -e "    └── Git Service (Gitea): ${DEPLOY_GITEA}"
    echo -e "  AI Memory (Mem0): SDK-based (uses existing PostgreSQL)"
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

setup_hyperdx_admin() {
    local admin_email="${HYPERDX_ADMIN_EMAIL:-marie@marie.local}"
    local admin_password="${HYPERDX_ADMIN_PASSWORD:-MarieAI@2026!}"
    local hyperdx_url="http://localhost:${HYPERDX_UI_PORT:-8080}"

    echo -e "${BLUE}👤 Setting up HyperDX admin user...${NC}"

    # Check if HyperDX is accessible
    if ! curl -sf "$hyperdx_url/" -o /dev/null 2>&1; then
        echo -e "${YELLOW}  ⚠️  HyperDX not ready, skipping user setup${NC}"
        return 1
    fi

    # Try to register user via API
    # HyperDX uses /api/register endpoint for user registration
    local register_response
    register_response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "{\"email\": \"$admin_email\", \"password\": \"$admin_password\"}" \
        "$hyperdx_url/api/register" 2>/dev/null)

    local http_code
    http_code=$(echo "$register_response" | tail -n1)
    local response_body
    response_body=$(echo "$register_response" | head -n -1)

    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        echo -e "${GREEN}  ✅ HyperDX admin user created${NC}"
        echo -e "${YELLOW}     Email: $admin_email${NC}"
        echo -e "${YELLOW}     Password: $admin_password${NC}"
    elif echo "$response_body" | grep -qi "already exists\|already registered\|email.*taken"; then
        echo -e "${GREEN}  ✅ HyperDX admin user already exists${NC}"
    else
        # Try alternative endpoint (some versions use different paths)
        register_response=$(curl -s -w "\n%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -d "{\"email\": \"$admin_email\", \"password\": \"$admin_password\", \"confirmPassword\": \"$admin_password\"}" \
            "$hyperdx_url/register" 2>/dev/null)

        http_code=$(echo "$register_response" | tail -n1)

        if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
            echo -e "${GREEN}  ✅ HyperDX admin user created${NC}"
            echo -e "${YELLOW}     Email: $admin_email${NC}"
            echo -e "${YELLOW}     Password: $admin_password${NC}"
        else
            echo -e "${YELLOW}  ⚠️  Could not auto-create HyperDX user${NC}"
            echo -e "${YELLOW}     Please create manually at: $hyperdx_url${NC}"
            echo -e "${YELLOW}     Suggested credentials:${NC}"
            echo -e "${YELLOW}       Email: $admin_email${NC}"
            echo -e "${YELLOW}       Password: $admin_password${NC}"
        fi
    fi
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

    # Create Mem0 database for SDK usage (always created, SDK-based integration)
    local mem0_db="${MEM0_DB_NAME:-mem0}"
    echo "  Creating database '$mem0_db' for Mem0 SDK..."
    docker exec marie-psql-server psql -U "${POSTGRES_USER:-postgres}" -tc \
        "SELECT 1 FROM pg_database WHERE datname = '$mem0_db'" | grep -q 1 || \
        docker exec marie-psql-server psql -U "${POSTGRES_USER:-postgres}" -c \
        "CREATE DATABASE $mem0_db" >/dev/null 2>&1
    # Enable pgvector extension for Mem0
    echo "  Enabling pgvector extension in '$mem0_db'..."
    docker exec marie-psql-server psql -U "${POSTGRES_USER:-postgres}" -d "$mem0_db" -c \
        "CREATE EXTENSION IF NOT EXISTS vector" >/dev/null 2>&1
    echo -e "${GREEN}  ✅ Database '$mem0_db' ready with pgvector${NC}"

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

            # Initialize schema (tables)
            local schema_file="./config/clickhouse/schema/llm_tracking.sql"
            if [ -f "$schema_file" ]; then
                echo "  Initializing ClickHouse schema..."
                docker exec -i marie-clickhouse clickhouse-client \
                    --database "$ch_db" \
                    < "$schema_file" >/dev/null 2>&1
                echo -e "${GREEN}  ✅ ClickHouse schema initialized (traces, observations, scores)${NC}"
            else
                echo -e "${YELLOW}  ⚠️  Schema file not found: $schema_file${NC}"
            fi
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

            # Gitea Setup Section - make output more visible
            echo ""
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}    Gitea Configuration${NC}"
            echo -e "${BLUE}========================================${NC}"

            # Create default admin user
            setup_gitea_admin

            # Create default repositories
            setup_gitea_repos

            # Create OAuth2 application for Marie Studio
            setup_gitea_oauth_app

            echo -e "${BLUE}========================================${NC}"
            echo ""
        fi

        # Start ClickStack (requires ClickHouse to be ready)
        if [ "$DEPLOY_CLICKSTACK" = "true" ] && [ "$DEPLOY_CLICKHOUSE" = "true" ] && [ -f "./Dockerfiles/docker-compose.clickstack.yml" ]; then
            echo -e "${YELLOW}⏳ Starting ClickStack (observability stack)...${NC}"

            # Initialize observability schema in ClickHouse
            local otel_schema_file="./config/clickstack/schema/observability.sql"
            if [ -f "$otel_schema_file" ]; then
                echo "  Initializing ClickHouse observability schema..."
                docker exec -i marie-clickhouse clickhouse-client \
                    < "$otel_schema_file" >/dev/null 2>&1 || true
                echo -e "${GREEN}  ✅ Observability schema initialized (otel_logs, otel_traces, otel_metrics)${NC}"
            fi

            # Start ClickStack services as part of infrastructure
            local clickstack_cmd="COMPOSE_NETWORK_MODE=host docker compose --env-file $ENV_FILE"
            clickstack_cmd="$clickstack_cmd --project-name marie-infrastructure"
            clickstack_cmd="$clickstack_cmd -f ./Dockerfiles/docker-compose.storage.yml"
            clickstack_cmd="$clickstack_cmd -f ./Dockerfiles/docker-compose.s3.yml"
            clickstack_cmd="$clickstack_cmd -f ./Dockerfiles/docker-compose.rabbitmq.yml"
            clickstack_cmd="$clickstack_cmd -f ./Dockerfiles/docker-compose.etcd.yml"
            clickstack_cmd="$clickstack_cmd -f ./Dockerfiles/docker-compose.clickhouse.yml"
            if [ "$DEPLOY_LITELLM" = "true" ]; then
                clickstack_cmd="$clickstack_cmd -f ./Dockerfiles/docker-compose.litellm.yml"
            fi
            if [ "$DEPLOY_GITEA" = "true" ]; then
                clickstack_cmd="$clickstack_cmd -f ./Dockerfiles/docker-compose.gitea.yml"
            fi
            clickstack_cmd="$clickstack_cmd -f ./Dockerfiles/docker-compose.clickstack.yml"
            clickstack_cmd="$clickstack_cmd --project-directory ."

            # Start ClickStack services
            eval "$clickstack_cmd up -d hyperdx log-collector"

            # Wait for HyperDX to be healthy (check container health status)
            echo -e "${YELLOW}⏳ Waiting for HyperDX to be healthy...${NC}"
            local hdx_attempts=30
            local hdx_attempt=1
            while [ $hdx_attempt -le $hdx_attempts ]; do
                # Check container health status
                local hdx_health
                hdx_health=$(docker inspect marie-hyperdx --format='{{.State.Health.Status}}' 2>/dev/null || echo "unknown")
                if [ "$hdx_health" = "healthy" ]; then
                    break
                fi
                # Also try checking if the UI responds (fallback)
                if curl -sf http://localhost:${HYPERDX_UI_PORT:-8080}/ -o /dev/null 2>&1; then
                    break
                fi
                echo "  Waiting for HyperDX (attempt $hdx_attempt/$hdx_attempts, status: $hdx_health)..."
                sleep 3
                ((hdx_attempt++))
            done

            if [ $hdx_attempt -gt $hdx_attempts ]; then
                echo -e "${YELLOW}⚠️  HyperDX may still be starting up (check: docker logs marie-hyperdx)${NC}"
            else
                echo -e "${GREEN}✅ ClickStack is ready${NC}"
            fi

            # ClickStack Setup Section (using hyperdx-local, no auth required)
            echo ""
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}    ClickStack Configuration${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo -e "${GREEN}  ✅ HyperDX Local mode - no authentication required${NC}"
            echo -e "${YELLOW}     UI: http://localhost:${HYPERDX_UI_PORT:-8080}${NC}"
            echo -e "${YELLOW}     OTLP HTTP: http://localhost:${OTEL_HTTP_PORT:-4318}${NC}"
            echo -e "${YELLOW}     OTLP gRPC: localhost:${OTEL_GRPC_PORT:-4317}${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo ""
        fi

        # Note: Mem0 is now integrated as a Python SDK, not a separate container
        # The mem0 database is created in initialize_databases() and the SDK uses it directly

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

        if [ "$DEPLOY_CLICKSTACK" = "true" ] && [ -f "./Dockerfiles/docker-compose.clickstack.yml" ]; then
            status_compose_cmd="$status_compose_cmd -f ./Dockerfiles/docker-compose.clickstack.yml"
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

        if [ "$DEPLOY_CLICKSTACK" = "true" ]; then
            echo "  📊 HyperDX UI: http://localhost:${HYPERDX_UI_PORT:-8080} (no auth required)"
            echo "  📊 OTLP gRPC: localhost:${OTEL_GRPC_PORT:-4317}"
            echo "  📊 OTLP HTTP: localhost:${OTEL_HTTP_PORT:-4318}"
        fi

        # Note: Mem0 is integrated as a Python SDK, not a container
        echo "  🧠 Mem0: SDK-based (uses PostgreSQL database 'mem0')"
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
            --vagrant)
                VAGRANT_MODE=true
                shift
                ;;
            --vagrant-up)
                VAGRANT_ACTION="up"
                shift
                ;;
            --vagrant-down)
                VAGRANT_ACTION="down"
                shift
                ;;
            --vagrant-ssh)
                VAGRANT_ACTION="ssh"
                shift
                ;;
            --vagrant-status)
                VAGRANT_ACTION="status"
                shift
                ;;
            --vagrant-list)
                vagrant_list
                exit 0
                ;;
            --instance=*)
                VAGRANT_INSTANCE="${1#*=}"
                if ! [[ "$VAGRANT_INSTANCE" =~ ^[1-9]$ ]]; then
                    echo -e "${RED}❌ Invalid instance ID: $VAGRANT_INSTANCE (must be 1-9)${NC}"
                    exit 1
                fi
                VAGRANT_VM_NAME="marie-test-${VAGRANT_INSTANCE}"
                shift
                ;;
            --sync-images)
                VAGRANT_SYNC_IMAGES=true
                shift
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
                DEPLOY_CLICKSTACK=false  # ClickStack requires ClickHouse
                shift
                ;;
            --no-clickstack)
                DEPLOY_CLICKSTACK=false
                shift
                ;;
            --with-clickstack)
                DEPLOY_CLICKSTACK=true
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
    echo "  --no-clickstack       Skip ClickStack observability stack (HyperDX + log collector)"
    echo "  --with-clickstack     Enable ClickStack observability stack (disabled by default)"
    echo "  --no-gitea            Skip Gitea Git service deployment"
    echo "  --infrastructure-only Deploy only infrastructure services"
    echo "  --services-only       Deploy only Marie application services (gateway + extract)"
    echo "  --litellm-only        Deploy only LiteLLM proxy (with required infrastructure)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Vagrant Options (isolated VM testing):"
    echo "  --vagrant             Run deployment inside Vagrant VM"
    echo "  --vagrant-up          Start/create Vagrant VM (without deploying)"
    echo "  --vagrant-down        Destroy Vagrant VM"
    echo "  --vagrant-ssh         SSH into the Vagrant VM"
    echo "  --vagrant-status      Show Vagrant VM status"
    echo "  --vagrant-list        List all Vagrant instances"
    echo "  --instance=N          Specify instance number (1-9, default: 1)"
    echo ""
    echo "Service Categories:"
    echo "  Infrastructure: Storage (MinIO), Message Queue (RabbitMQ), Service Discovery (etcd),"
    echo "                  LLM Proxy (LiteLLM), Analytics DB (ClickHouse), Observability (ClickStack),"
    echo "                  Git Service (Gitea)"
    echo "  AI Memory:      Mem0 (SDK-based, uses existing PostgreSQL)"
    echo "  Application:    Gateway, Extract Executors"
    echo ""
    echo "Examples:"
    echo "  $0                              # Deploy everything (ClickStack disabled by default)"
    echo "  $0 --with-clickstack            # Deploy everything including observability"
    echo "  $0 --stop-all                   # Stop all services and cleanup"
    echo "  $0 --infrastructure-only        # Deploy infrastructure only"
    echo "  $0 --services-only              # Deploy only gateway + extract"
    echo "  $0 --no-extract                 # Deploy infrastructure + gateway only"
    echo "  $0 --no-clickhouse --no-gitea   # Deploy without analytics and Git"
    echo "  $0 --litellm-only               # Deploy minimal infrastructure + LiteLLM"
    echo ""
    echo "Vagrant Examples:"
    echo "  $0 --vagrant                    # Deploy full stack in instance 1"
    echo "  $0 --vagrant --instance=2       # Deploy in instance 2"
    echo "  $0 --vagrant --infrastructure-only  # Deploy only infrastructure in VM"
    echo "  $0 --vagrant --sync-images      # Sync local Docker images to VM first"
    echo "  $0 --vagrant-up                 # Start instance 1"
    echo "  $0 --vagrant-up --instance=2    # Start instance 2"
    echo "  $0 --vagrant-ssh --instance=2   # SSH into instance 2"
    echo "  $0 --vagrant-status --instance=2    # Check instance 2 status"
    echo "  $0 --vagrant-list               # List all instances"
    echo "  $0 --vagrant-down --instance=2  # Destroy instance 2"
    echo ""
    echo "Multi-Instance Port Mapping:"
    echo "  Instance 1: PostgreSQL=15432, ClickHouse=18123, RabbitMQ=25672"
    echo "  Instance 2: PostgreSQL=16432, ClickHouse=19123, RabbitMQ=26672"
    echo "  Instance 3: PostgreSQL=17432, ClickHouse=20123, RabbitMQ=27672"
}

main() {
    parse_args "$@"

    # Handle Vagrant actions (must be after parse_args to get --instance)
    if [ -n "$VAGRANT_ACTION" ]; then
        case "$VAGRANT_ACTION" in
            up)
                vagrant_up
                exit 0
                ;;
            down)
                vagrant_down
                exit 0
                ;;
            ssh)
                vagrant_ssh
                exit 0
                ;;
            status)
                vagrant_status
                exit 0
                ;;
        esac
    fi

    # Handle Vagrant deployment mode
    if [ "$VAGRANT_MODE" = "true" ]; then
        vagrant_bootstrap
        exit $?
    fi

    # Standard bare-metal deployment
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
    if [ "$DEPLOY_CLICKSTACK" = "true" ]; then
        echo "  View HyperDX logs: docker logs -f marie-hyperdx"
        echo "  View Log Collector logs: docker logs -f marie-log-collector"
        echo "  HyperDX UI: http://localhost:8080"
    fi
    echo ""
    echo "Mem0 (SDK-based):"
    echo "  Database: mem0 (PostgreSQL with pgvector)"
    echo "  Usage: pip install 'marie-ai[memory]'"
}

main "$@"