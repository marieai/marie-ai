# Python Wasm Component Compiler
# Compiles Python source to WebAssembly Component Model modules

FROM python:3.13-slim-bookworm

# Install componentize-py (latest version as of Jan 2026)
RUN pip install --no-cache-dir componentize-py==0.19.3

# Create non-root user for security (use UID 1001 to avoid conflicts)
RUN useradd -m -u 1001 compiler && \
    mkdir -p /opt/wit /workspace && \
    chown -R compiler:compiler /opt /workspace

# Copy WIT definitions
COPY --chown=compiler:compiler wit/ /opt/wit/

# Copy compile script
COPY --chown=compiler:compiler python/compile.sh /opt/compile.sh
RUN chmod +x /opt/compile.sh

# Switch to non-root user
USER compiler

WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "/opt/compile.sh"]
