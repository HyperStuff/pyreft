#!/bin/bash
set -e  # Exit on error

# Print system information
echo "🖥️  Container System Information:"
nvidia-smi
echo "-----------------------------------"

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA is available"
    # Get number of available GPUs
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "📊 Found ${GPU_COUNT} GPU(s)"
    # Set CUDA_VISIBLE_DEVICES to all available GPUs (0,1,2,etc.)
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((GPU_COUNT-1)))
    echo "🎯 CUDA_VISIBLE_DEVICES set to: ${CUDA_VISIBLE_DEVICES}"
else
    echo "⚠️  WARNING: CUDA is not available"
fi

# Print Python environment information
echo "🐍 Python Environment Information:"
python --version
pip list

# Print working directory information
echo "📂 Current working directory: $(pwd)"
echo "📂 Contents of current directory:"
ls -la

# Move git config setup to beginning before any other operations
echo "🔧 Setting up git configuration..."

git config --global --replace-all user.email "${GIT_EMAIL}"
git config --global --replace-all user.name "${GIT_NAME}"
git config --global --replace-all safe.directory /workspace/${PROJECT_ROOT}

echo "🚀 Container is ready!"
echo "-----------------------------------"

# Execute the command passed to docker run
echo "🔄 Executing command: $@"
exec "$@"