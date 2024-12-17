ARG GIT_NAME
ARG GIT_EMAIL
ARG PROJECT_NAME

FROM ghcr.io/${GIT_NAME}/ml-base:latest

WORKDIR /workspace/${PROJECT_NAME}

# Project-specific requirements only
COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system --no-cache-dir -r requirements.txt

COPY scripts/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]