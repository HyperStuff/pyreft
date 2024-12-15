FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ARG GIT_EMAIL
ARG GIT_NAME

WORKDIR /workspace/pyreft

# Combine environment variable settings
ENV TZ=Etc/UTC \
    PATH="/usr/local/bin:${PATH}" \
    PYTHONPATH="/usr/local/lib/python3.12/dist-packages:${PYTHONPATH}" \
    GIT_EMAIL=${GIT_EMAIL} \
    GIT_NAME=${GIT_NAME}

# Set timezone in the same layer as other operations
# Combine all apt operations and cleanup
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libdrm-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libsystemd-dev \
    libsystemd0 \
    libudev-dev \
    libudev0 \
    nano \
    ncdu \
    nvtop \
    openssh-client \
    software-properties-common \
    screen \
    sudo \
    vim \
    wget \
    zsh && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3.12-venv && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    git config --global core.editor "vim" && \
    git config --global user.email "${GIT_EMAIL}" && \
    git config --global user.name "${GIT_NAME}" && \
    echo "set -o vi" >> ~/.bashrc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements only once and install dependencies
COPY requirements.txt .
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install --system --no-cache-dir --upgrade pip setuptools wheel && \
    /root/.local/bin/uv pip install --system --no-cache-dir -r requirements.txt

# Set up entrypoint
COPY scripts/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

SHELL ["/bin/zsh", "-c"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]