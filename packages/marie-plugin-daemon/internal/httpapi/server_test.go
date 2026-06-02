<<<<<<< HEAD
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
	"time"

	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/auth"
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

func TestRuntimeInvocationRejectsUnsignedEnvelope(t *testing.T) {
	server := NewServer(VersionInfo{Version: "test", Commit: "abc", Mode: "decode_only"})
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/runtime/stub-invocations", bytes.NewReader([]byte(`{}`))))
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", response.Code)
	}
}

func TestRuntimeInvocationUnsupportedAfterEnvelopeVerification(t *testing.T) {
	now := time.Date(2026, 6, 2, 12, 0, 0, 0, time.UTC)
	verifier := auth.NewEnvelopeVerifier([]auth.SigningKey{{KeyID: "marie-api-test", Secret: "test-runtime-secret"}}, func() time.Time {
		return now
	})
	server := NewServer(VersionInfo{Version: "test", Commit: "abc", Mode: "decode_only"}, WithEnvelopeVerifier(verifier))
	envelope := signedEnvelope("nonce-1", now.Add(5*time.Minute))
	body, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}

	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/runtime/stub-invocations", bytes.NewReader(body)))
	if response.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d", response.Code)
	}
}

func TestRuntimeInvocationRejectsPolicyDeniedEnvelope(t *testing.T) {
	now := time.Date(2026, 6, 2, 12, 0, 0, 0, time.UTC)
	verifier := auth.NewEnvelopeVerifier([]auth.SigningKey{{KeyID: "marie-api-test", Secret: "test-runtime-secret"}}, func() time.Time {
		return now
	})
	server := NewServer(VersionInfo{Version: "test", Commit: "abc", Mode: "decode_only"}, WithEnvelopeVerifier(verifier))
	envelope := signedEnvelope("nonce-policy", now.Add(5*time.Minute))
	envelope["packageTrustLevel"] = "blocked"
	envelope["signature"] = map[string]any{
		"keyId":     "marie-api-test",
		"algorithm": "hmac-sha256",
		"value":     signEnvelope(envelope, "test-runtime-secret"),
	}

	body, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/runtime/stub-invocations", bytes.NewReader(body)))
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

func signedEnvelope(nonce string, expiresAt time.Time) map[string]any {
	envelope := map[string]any{
		"requestId":            "request-1",
		"traceId":              "trace-1",
		"organizationId":       "11111111-1111-1111-1111-111111111111",
		"workspaceId":          "22222222-2222-2222-2222-222222222222",
		"userId":               "44444444-4444-4444-4444-444444444444",
		"installId":            "33333333-3333-3333-3333-333333333333",
		"packageId":            "77777777-7777-7777-7777-777777777777",
		"packageRef":           "ext.langgenius.search",
		"packageDigest":        "sha256:package",
		"packageTrustLevel":    "community",
		"providerId":           "55555555-5555-5555-5555-555555555555",
		"actionId":             "tools/search",
		"actionType":           "stub",
		"credentialBindingIds": []any{},
		"input":                map[string]any{"query": "invoices"},
		"runtimePolicy":        map[string]any{"timeoutMs": float64(30000), "maxConcurrent": float64(1), "maxMemoryBytes": float64(536870912), "networkPolicy": "none"},
		"expiresAt":            expiresAt.Format(time.RFC3339),
		"nonce":                nonce,
		"mode":                 "stub",
	}
	envelope["signature"] = map[string]any{
		"keyId":     "marie-api-test",
		"algorithm": "hmac-sha256",
		"value":     signEnvelope(envelope, "test-runtime-secret"),
	}
	return envelope
}

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
||||||| 34767ebc
=======
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
>>>>>>> ffc574d398b2874e2ae5244ba61a602382254b37
