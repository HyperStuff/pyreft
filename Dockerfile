ARG GIT_NAME
ARG GIT_EMAIL
ARG PROJECT_NAME

FROM ghcr.io/${GIT_NAME}/ml-base:latest

WORKDIR /workspace/${PROJECT_NAME}

# Project-specific requirements only
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# If python package exists, install it
RUN python setup.py develop || true \
    && pip install --no-cache-dir -e . || true

COPY scripts/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]