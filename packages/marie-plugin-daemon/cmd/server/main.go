package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel"
	backwards_invocation "github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel/backwards_invocation"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/plugin_manager"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/httpapi"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/auth"
)

func main() {
	defaultRoot := os.Getenv("MARIE_PLUGIN_STORAGE_ROOT")
	if defaultRoot == "" {
		defaultRoot = "./storage"
	}

	addr := flag.String("addr", "127.0.0.1:8099", "HTTP listen address")
	storageRoot := flag.String("storage-root", defaultRoot, "plugin storage root directory")
	flag.Parse()

	manager := plugin_manager.NewManager(*storageRoot)
	pool := io_tunnel.NewPool(manager, backwards_invocation.NewStorage(*storageRoot), os.Stderr)

	options := []httpapi.ServerOption{
		httpapi.WithManager(manager),
		httpapi.WithPool(pool),
	}
	keyID := os.Getenv("MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID")
	secret := os.Getenv("MARIE_PLUGIN_DAEMON_SIGNING_SECRET")
	if keyID != "" && secret != "" {
		verifier := auth.NewEnvelopeVerifier([]auth.SigningKey{{KeyID: keyID, Secret: secret}}, nil)
		options = append(options, httpapi.WithEnvelopeVerifier(verifier))
	} else {
		log.Print("warning: MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID/SECRET not set, signed routes will reject all requests")
	}

	handler := httpapi.NewServer(httpapi.VersionInfo{
		Version: "0.1.0",
		Commit:  "unknown",
		Mode:    "runtime",
	}, options...)

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
		log.Print("shutting down: draining HTTP, then stopping plugin instances")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			log.Printf("http shutdown: %v", err)
		}
		pool.Shutdown()
		close(shutdownDone)
	}()

	log.Printf("marie-plugin-daemon listening on %s (storage root %s)", *addr, *storageRoot)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	<-shutdownDone
}
