package httpapi

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
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

func TestLegacyStubInvocationRouteIsRemoved(t *testing.T) {
	server := newRuntimeServer(t)
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/runtime/stub-invocations", bytes.NewReader([]byte(`{}`))))
	if response.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for removed stub route, got %d", response.Code)
	}
}

func TestDispatchInvocationRejectsUnsignedEnvelope(t *testing.T) {
	server := newRuntimeServer(t)
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/dispatch/invoke", bytes.NewReader([]byte(`{}`))))
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", response.Code)
	}
}

func TestDispatchInvocationRejectsPolicyDeniedEnvelope(t *testing.T) {
	server := newRuntimeServer(t)
	envelope := runtimeEnvelope(t, map[string]any{"packageTrustLevel": "blocked"})

	body, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/dispatch/invoke", bytes.NewReader(body)))
	if response.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", response.Code, response.Body.String())
	}
	if !bytes.Contains(response.Body.Bytes(), []byte("trust_policy_denied")) {
		t.Fatalf("expected trust denial body, got %s", response.Body.String())
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

func signEnvelope(envelope map[string]any, secret string) string {
	payload := map[string]any{}
	for key, value := range envelope {
		if key != "signature" {
			payload[key] = value
		}
	}
	canonical, err := json.Marshal(payload)
	if err != nil {
		panic(err)
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(canonical)
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}
