#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/setup_remote.sh <instance_ip>"
    exit 1
fi

REMOTE_IP=$1
LOCAL_ROOT=$(pwd)  # Local project root
PROJECT_NAME=$(basename "${LOCAL_ROOT}")  # Just the last component
REMOTE_ROOT="/home/ubuntu/projects/${PROJECT_NAME}"  # Simple remote path

setup_ssh_tunnel() {
    local ip=$1
    
    echo "🔍 Checking for existing tunnels..."
    # Using new ports
    lsof -ti:8765 | xargs kill -9 2>/dev/null || true
    lsof -ti:6380 | xargs kill -9 2>/dev/null || true
    lsof -ti:10001 | xargs kill -9 2>/dev/null || true
    
    echo "🔗 Creating new SSH tunnel..."
    # Updated port mappings
    ssh -N -L 8765:localhost:8765 -L 6380:localhost:6380 -L 10001:localhost:10001 ubuntu@$ip &
    
    echo $! > /tmp/remote_tunnel.pid
    sleep 2
    
    echo "✅ SSH tunnel established. Ray dashboard available at http://localhost:8765"
}

if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "📝 Loading local environment variables..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "🔄 Setting up instance..."

# Create directories
ssh ubuntu@$REMOTE_IP "sudo mkdir -p ${REMOTE_ROOT} && sudo chown ubuntu:ubuntu ${REMOTE_ROOT}"

# Set correct permissions before syncing
echo "🔒 Setting correct SSH permissions..."
ssh ubuntu@$REMOTE_IP "mkdir -p ~/.ssh && chmod 700 ~/.ssh"

# Sync credentials with correct permissions
echo "🔑 Syncing credentials..."
rsync -avz -e ssh \
    ~/.ssh/github* \
    ubuntu@$REMOTE_IP:~/.ssh/

# Add CLUSTER_IP to local .env if not already present
if ! grep -q "^CLUSTER_IP=" "${LOCAL_ROOT}/.env"; then
    echo "📝 Adding CLUSTER_IP to local .env..."
    echo "CLUSTER_IP=$REMOTE_IP" >> "${LOCAL_ROOT}/.env"
    echo "✅ Added CLUSTER_IP=$REMOTE_IP to local environment"
fi

# Sync environment file
echo "📄 Syncing .env file..."
rsync -avz -e ssh \
    "${LOCAL_ROOT}/.env" \
    "ubuntu@${REMOTE_IP}:${REMOTE_ROOT}/"

# Sync code
echo "📦 Syncing code..."
rsync -avz --progress -e ssh \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'assets' \
    --exclude '.env' \
    --exclude 'node_modules' \
    --exclude '.venv' \
    "${LOCAL_ROOT}/" \
    "ubuntu@${REMOTE_IP}:${REMOTE_ROOT}/"

echo "🔧 Making scripts executable..."
ssh ubuntu@$REMOTE_IP "cd ${REMOTE_ROOT} && chmod +x scripts/*.sh"

echo "🔗 Setting up SSH tunnel..."
setup_ssh_tunnel $REMOTE_IP

echo "🚀 Starting container on Lambda instance..."
ssh -t ubuntu@$REMOTE_IP "cd ${REMOTE_ROOT} && ./scripts/container_setup.sh"