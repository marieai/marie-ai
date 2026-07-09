package io_tunnel

import (
	"context"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// wasmFixtureZip packages testdata/fixture-wasm-plugin (a python_source plugin
// whose entrypoint is marie_wasm.daemon_runner + a compiled node.wasm).
func wasmFixtureZip(t *testing.T) []byte {
	t.Helper()
	root := filepath.Join("..", "..", "..", "testdata", "fixture-wasm-plugin")
	files := map[string][]byte{}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		files[filepath.ToSlash(rel)] = data
		return nil
	})
	if err != nil {
		t.Fatalf("read wasm fixture: %v", err)
	}
	return zipFromFiles(t, files)
}

// TestWasmNodeRunsThroughDaemon is the W1 acceptance: the built-in http-request
// WASM node runs as a daemon-managed plugin. The daemon builds the venv
// (installs marie-wasm), spawns `python -m marie_wasm.daemon_runner`, which
// loads node.wasm, wires the host imports, and on invoke makes a real outbound
// HTTP call through the Go-less Python host, streaming the result back.
func TestWasmNodeRunsThroughDaemon(t *testing.T) {
	requireUV(t)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer upstream.Close()

	pool, install := newDeployedPool(t, wasmFixtureZip(t))

	payload := map[string]any{
		"input": []any{},
		"env":   `{"method":"GET","url":"` + upstream.URL + `","headers":{}}`,
		"ctx":   map[string]any{"workflow_id": "wf", "execution_id": "ex", "node_id": "n1", "run_index": 0},
	}
	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, payload, 60*time.Second)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}
	collected := collectFrames(t, frames, 60*time.Second)

	var sawStream200, sawEnd bool
	for _, f := range collected {
		switch f.Type {
		case FrameStream:
			d := string(f.Data)
			if strings.Contains(d, `\"status\":200`) || strings.Contains(d, `"status":200`) {
				sawStream200 = true
			}
		case FrameEnd:
			sawEnd = true
		case FrameError:
			t.Fatalf("error frame: %s", f.Data)
		}
	}
	if !sawStream200 || !sawEnd {
		t.Fatalf("expected stream(status 200)+end, got %+v", collected)
	}
}
