# Rust Wasm Component Compiler
# Compiles Rust source to WebAssembly Component Model modules

FROM rust:1.93-bookworm

# Install cargo-component for component model support
RUN cargo install cargo-component --locked

# Install wasm32-wasip1 target (renamed from wasm32-wasi in newer Rust) and rustfmt
RUN rustup target add wasm32-wasip1 && \
    rustup component add rustfmt

# Create non-root user for security (use UID 1001 to avoid conflicts)
RUN useradd -m -u 1001 compiler && \
    mkdir -p /opt/wit /workspace && \
    chown -R compiler:compiler /opt /workspace

# Copy WIT definitions
COPY --chown=compiler:compiler wit/ /opt/wit/

# Copy compile script
COPY --chown=compiler:compiler rust/compile.sh /opt/compile.sh
RUN chmod +x /opt/compile.sh

# Switch to non-root user
USER compiler

WORKDIR /workspace

# Set cargo home to writable location
ENV CARGO_HOME=/workspace/.cargo
ENV PATH="/workspace/.cargo/bin:${PATH}"

ENTRYPOINT ["/opt/compile.sh"]
