package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/httpapi"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8099", "HTTP listen address")
	flag.Parse()

	server := httpapi.NewServer(httpapi.VersionInfo{
		Version: "0.1.0-decode",
		Commit:  "unknown",
		Mode:    "decode_only",
	})

	log.Printf("marie-plugin-daemon listening on %s", *addr)
	if err := http.ListenAndServe(*addr, server); err != nil {
		log.Fatal(err)
	}
}
