package httpapi

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestHealthAndDecode(t *testing.T) {
	root := t.TempDir()
	manifestPath := filepath.Join(root, "marie-extension.yaml")
	if err := os.WriteFile(manifestPath, []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}

	server := NewServer(VersionInfo{Version: "test", Commit: "abc", Mode: "decode_only"})

	health := httptest.NewRecorder()
	server.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/health", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("health returned %d", health.Code)
	}

	body, err := json.Marshal(map[string]string{"path": root})
	if err != nil {
		t.Fatal(err)
	}
	decode := httptest.NewRecorder()
	server.ServeHTTP(decode, httptest.NewRequest(http.MethodPost, "/v1/packages/decode", bytes.NewReader(body)))
	if decode.Code != http.StatusOK {
		t.Fatalf("decode returned %d: %s", decode.Code, decode.Body.String())
	}
}

func TestRuntimeInvocationUnsupported(t *testing.T) {
	server := NewServer(VersionInfo{Version: "test", Commit: "abc", Mode: "decode_only"})
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/runtime/stub-invocations", nil))
	if response.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d", response.Code)
	}
}

const manifest = `apiVersion: marie.ai/v1alpha1
kind: ExtensionPackage
metadata:
  id: ext.test.minimal-tool
  name: minimal-tool
  version: 0.1.0
providers:
  - ref: provider/minimal
    type: tool_provider
`
