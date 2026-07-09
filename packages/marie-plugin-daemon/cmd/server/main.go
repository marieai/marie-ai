package main

import (
	"context"
	"errors"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/config"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel"
	backwards_invocation "github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel/backwards_invocation"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/plugin_manager"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/httpapi"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/auth"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/pkg/utils/log"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		// Logger not initialized yet; fail loudly via panic (stderr).
		panic("error processing environment variables: " + err.Error())
	}

	logCloser, err := log.Init(cfg.LogJSON(), cfg.LogFile, cfg.LogLevel)
	if err != nil {
		panic("failed to init logger: " + err.Error())
	}
	if logCloser != nil {
		defer func() { _ = logCloser.Close() }()
	}
	defer log.RecoverAndExit()

	// Flags override the env-provided defaults (back-compat with -addr/-storage-root).
	addr := flag.String("addr", cfg.Addr, "HTTP listen address")
	storageRoot := flag.String("storage-root", cfg.StorageRoot, "plugin storage root directory")
	flag.Parse()

	manager := plugin_manager.NewManager(*storageRoot)
	pool := io_tunnel.NewPool(manager, backwards_invocation.NewStorage(*storageRoot), os.Stderr)

	options := []httpapi.ServerOption{
		httpapi.WithManager(manager),
		httpapi.WithPool(pool),
	}
	switch {
	case bool(cfg.DevInsecure):
		log.Warn("MARIE_PLUGIN_DAEMON_DEV_INSECURE is set — envelope signatures are NOT verified (DEV ONLY, do not use in shared environments)")
		options = append(options, httpapi.WithEnvelopeVerifier(auth.NewInsecureEnvelopeVerifier()))
	case cfg.SigningConfigured():
		verifier := auth.NewEnvelopeVerifier([]auth.SigningKey{{KeyID: cfg.SigningKeyID, Secret: cfg.SigningSecret}}, nil)
		options = append(options, httpapi.WithEnvelopeVerifier(verifier))
	default:
		log.Warn("MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID/SECRET not set, signed routes will reject all requests")
	}

	var handler http.Handler = httpapi.NewServer(httpapi.VersionInfo{
		Version: "0.1.0",
		Commit:  "unknown",
		Mode:    "runtime",
	}, options...)

	// Logging middleware: recover panics (outermost) -> attach trace -> log request.
	handler = log.Chain(handler,
		log.RecoveryMiddleware,
		log.TraceMiddleware,
		log.LoggerMiddleware("/health"),
	)

	server := &http.Server{
		Addr:              *addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		// WriteTimeout stays 0: SSE invocations stream for their full runtime.
	}

	shutdownDone := make(chan struct{})
	go func() {
		signals := make(chan os.Signal, 1)
		signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
		<-signals
		log.Info("shutting down: draining HTTP, then stopping plugin instances")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			log.Error("http shutdown error", "error", err)
		}
		pool.Shutdown()
		close(shutdownDone)
	}()

	log.Info("marie-plugin-daemon listening", "addr", *addr, "storage_root", *storageRoot)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Panic("server error", "error", err)
	}
	<-shutdownDone
}
