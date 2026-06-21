FROM node:24-bookworm-slim

WORKDIR /app

ENV AIMOCK_HOST=0.0.0.0

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies with enough output to diagnose slow registry/package steps.
RUN npm ci --loglevel=verbose --foreground-scripts --timing

# Copy source files
COPY *.ts *.mjs ./

# Expose the default port
EXPOSE 4010 4011

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD ["node", "healthcheck.mjs"]

# Run the server
CMD ["npm", "start"]
