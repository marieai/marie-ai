FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm install

# Copy source files
COPY *.ts ./

# Expose the default port
EXPOSE 4010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:${AIMOCK_PORT:-4010}/health || exit 1

# Run the server
CMD ["npm", "start"]
