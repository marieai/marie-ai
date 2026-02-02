# JavaScript/TypeScript Wasm Component Compiler
# Compiles JavaScript or TypeScript source to WebAssembly Component Model modules

FROM node:24-bookworm-slim

# Install jco, componentize-js, esbuild, and typescript (latest versions as of Jan 2026)
RUN npm install -g \
    @bytecodealliance/jco@1.15.4 \
    @bytecodealliance/componentize-js@0.18.4 \
    esbuild@0.27.2 \
    typescript@5.9

# Use existing node user (UID 1000) and create directories
RUN mkdir -p /opt/wit /opt/types /workspace && \
    chown -R node:node /opt /workspace

# Copy WIT definitions
COPY --chown=node:node wit/ /opt/wit/

# Copy TypeScript type definitions
COPY --chown=node:node js/types/ /opt/types/

# Copy compile script
COPY --chown=node:node js/compile.sh /opt/compile.sh
RUN chmod +x /opt/compile.sh

# Switch to non-root user
USER node

WORKDIR /workspace

# Set npm cache to writable location
ENV npm_config_cache=/workspace/.npm

ENTRYPOINT ["/bin/bash", "/opt/compile.sh"]
