package httpapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel"
	backwards_invocation "github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel/backwards_invocation"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/plugin_manager"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/auth"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/decoder"
)

const fixturePackageRef = "ext.test.fixture-echo"

var envelopeNonce atomic.Int64

func newRuntimeServer(t *testing.T) http.Handler {
	t.Helper()
	root := t.TempDir()
	manager := plugin_manager.NewManager(root)
	pool := io_tunnel.NewPool(manager, backwards_invocation.NewStorage(root), io.Discard)
	t.Cleanup(pool.Shutdown)

	verifier := auth.NewEnvelopeVerifier([]auth.SigningKey{{KeyID: "marie-api-test", Secret: "test-runtime-secret"}}, nil)
	return NewServer(
		VersionInfo{Version: "test", Commit: "abc", Mode: "runtime"},
		WithEnvelopeVerifier(verifier),
		WithManager(manager),
		WithPool(pool),
	)
}

func requireUV(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
}

func runtimeEnvelope(t *testing.T, overrides map[string]any) map[string]any {
	t.Helper()
	envelope := map[string]any{
		"requestId":            "request-1",
		"traceId":              "trace-1",
		"organizationId":       "org1",
		"workspaceId":          "ws1",
		"userId":               "44444444-4444-4444-4444-444444444444",
		"installId":            "33333333-3333-3333-3333-333333333333",
		"packageId":            "77777777-7777-7777-7777-777777777777",
		"packageRef":           fixturePackageRef,
		"packageDigest":        "sha256:package",
		"packageTrustLevel":    "community",
		"providerId":           "55555555-5555-5555-5555-555555555555",
		"actionId":             "tools/echo",
		"actionType":           "stub",
		"credentialBindingIds": []any{},
		"input":                map[string]any{},
		"runtimePolicy":        map[string]any{"timeoutMs": float64(30000), "maxConcurrent": float64(1), "maxMemoryBytes": float64(536870912), "networkPolicy": "none"},
		"expiresAt":            time.Now().Add(5 * time.Minute).Format(time.RFC3339),
		"nonce":                fmt.Sprintf("nonce-%d-%d", time.Now().UnixNano(), envelopeNonce.Add(1)),
		"mode":                 "stub",
	}
	for key, value := range overrides {
		envelope[key] = value
	}
	envelope["signature"] = map[string]any{
		"keyId":     "marie-api-test",
		"algorithm": "hmac-sha256",
		"value":     signEnvelope(envelope, "test-runtime-secret"),
	}
	return envelope
}

func envelopeHeader(t *testing.T, envelope map[string]any) string {
	t.Helper()
	raw, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func fixtureArchive(t *testing.T) []byte {
	t.Helper()
	root := filepath.Join("..", "..", "testdata", "fixture-plugin")
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
		t.Fatalf("read fixture plugin: %v", err)
	}
	return zipArchive(t, files)
}

func zipArchive(t *testing.T, files map[string][]byte) []byte {
	t.Helper()
	archive, err := decoder.ZipFixture(files)
	if err != nil {
		t.Fatalf("zip fixture: %v", err)
	}
	return archive
}

// archiveIdentity decodes the archive the same way the daemon does, so test
// envelopes carry claims that match the upload (required since I4).
func archiveIdentity(t *testing.T, archive []byte) (packageRef, digest string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "pkg.zip")
	if err := os.WriteFile(path, archive, 0o644); err != nil {
		t.Fatal(err)
	}
	decoded, err := decoder.DecodePath(path)
	if err != nil {
		t.Fatalf("decode archive: %v", err)
	}
	return decoded.Identity.PackageRef, decoded.Checksum
}

func installEnvelope(t *testing.T, archive []byte, overrides map[string]any) map[string]any {
	t.Helper()
	packageRef, digest := archiveIdentity(t, archive)
	merged := map[string]any{"packageRef": packageRef, "packageDigest": digest}
	for key, value := range overrides {
		merged[key] = value
	}
	return runtimeEnvelope(t, merged)
}

func installArchive(t *testing.T, server http.Handler, archive []byte) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(http.MethodPost, "/v1/plugins/install", bytes.NewReader(archive))
	request.Header.Set("X-Marie-Envelope", envelopeHeader(t, installEnvelope(t, archive, nil)))
	response := httptest.NewRecorder()
	server.ServeHTTP(response, request)
	return response
}

func TestInstallListInvokeRemoveLifecycle(t *testing.T) {
	requireUV(t)
	server := newRuntimeServer(t)

	install := installArchive(t, server, fixtureArchive(t))
	if install.Code != http.StatusOK {
		t.Fatalf("install returned %d: %s", install.Code, install.Body.String())
	}
	if !strings.Contains(install.Body.String(), `"state":"ready"`) {
		t.Fatalf("expected ready state, got %s", install.Body.String())
	}

	list := httptest.NewRecorder()
	listRequest := httptest.NewRequest(http.MethodGet, "/v1/plugins", nil)
	listRequest.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, nil)))
	server.ServeHTTP(list, listRequest)
	if list.Code != http.StatusOK {
		t.Fatalf("list returned %d: %s", list.Code, list.Body.String())
	}
	if !strings.Contains(list.Body.String(), fixturePackageRef) || !strings.Contains(list.Body.String(), `"state":"ready"`) {
		t.Fatalf("expected ready install in list, got %s", list.Body.String())
	}

	invokeEnvelope := runtimeEnvelope(t, map[string]any{"payload": map[string]any{"ping": true}})
	invokeBody, err := json.Marshal(invokeEnvelope)
	if err != nil {
		t.Fatal(err)
	}
	invoke := httptest.NewRecorder()
	server.ServeHTTP(invoke, httptest.NewRequest(http.MethodPost, "/v1/dispatch/invoke", bytes.NewReader(invokeBody)))
	if invoke.Code != http.StatusOK {
		t.Fatalf("invoke returned %d: %s", invoke.Code, invoke.Body.String())
	}
	if contentType := invoke.Header().Get("Content-Type"); contentType != "text/event-stream" {
		t.Fatalf("expected SSE content type, got %q", contentType)
	}
	body := invoke.Body.String()
	if !strings.Contains(body, `"type":"stream"`) || !strings.Contains(body, `"echo"`) ||
		(!strings.Contains(body, `"event": "request"`) && !strings.Contains(body, `"event":"request"`)) {
		t.Fatalf("expected echoed stream frame, got %s", body)
	}
	if !strings.Contains(body, `"type":"end"`) {
		t.Fatalf("expected end frame, got %s", body)
	}
	if !strings.HasPrefix(body, "data: ") || !strings.Contains(body, "\n\n") {
		t.Fatalf("expected SSE framing, got %q", body)
	}

	remove := httptest.NewRecorder()
	removeRequest := httptest.NewRequest(http.MethodDelete, "/v1/plugins/"+fixturePackageRef, nil)
	removeRequest.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, nil)))
	server.ServeHTTP(remove, removeRequest)
	if remove.Code != http.StatusOK {
		t.Fatalf("remove returned %d: %s", remove.Code, remove.Body.String())
	}

	listAfter := httptest.NewRecorder()
	listAfterRequest := httptest.NewRequest(http.MethodGet, "/v1/plugins", nil)
	listAfterRequest.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, nil)))
	server.ServeHTTP(listAfter, listAfterRequest)
	if !strings.Contains(listAfter.Body.String(), `"plugins":[]`) {
		t.Fatalf("expected empty plugin list, got %s", listAfter.Body.String())
	}
}

func TestInvokeStorageRoundTripSSE(t *testing.T) {
	requireUV(t)
	server := newRuntimeServer(t)

	if response := installArchive(t, server, fixtureArchive(t)); response.Code != http.StatusOK {
		t.Fatalf("install returned %d: %s", response.Code, response.Body.String())
	}

	invoke := invokeRef(t, server, fixturePackageRef, map[string]any{
		"storage_roundtrip": true,
		"key":               "checkpoint",
		"value":             "v1",
	})
	if invoke.Code != http.StatusOK {
		t.Fatalf("invoke returned %d: %s", invoke.Code, invoke.Body.String())
	}
	if contentType := invoke.Header().Get("Content-Type"); contentType != "text/event-stream" {
		t.Fatalf("expected SSE content type, got %q", contentType)
	}
	body := invoke.Body.String()
	if !strings.Contains(body, `"type":"log"`) || !strings.Contains(body, "storage backwards invocation") {
		t.Fatalf("expected storage op log frames, got %s", body)
	}
	if !strings.Contains(body, `"storage_get":"v1"`) {
		t.Fatalf("expected GET round-trip to return v1, got %s", body)
	}
	if !strings.Contains(body, `"type":"end"`) {
		t.Fatalf("expected end frame, got %s", body)
	}
}

func TestInstallRejectsBadEnvelope(t *testing.T) {
	server := newRuntimeServer(t)

	missing := httptest.NewRecorder()
	server.ServeHTTP(missing, httptest.NewRequest(http.MethodPost, "/v1/plugins/install", bytes.NewReader([]byte("zip"))))
	if missing.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for missing envelope, got %d", missing.Code)
	}

	envelope := runtimeEnvelope(t, nil)
	envelope["packageRef"] = "ext.test.tampered"
	request := httptest.NewRequest(http.MethodPost, "/v1/plugins/install", bytes.NewReader([]byte("zip")))
	request.Header.Set("X-Marie-Envelope", envelopeHeader(t, envelope))
	tampered := httptest.NewRecorder()
	server.ServeHTTP(tampered, request)
	if tampered.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for tampered envelope, got %d: %s", tampered.Code, tampered.Body.String())
	}
}

func TestInvokeUnknownPlugin(t *testing.T) {
	server := newRuntimeServer(t)

	body, err := json.Marshal(runtimeEnvelope(t, map[string]any{"packageRef": "ext.test.missing"}))
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/dispatch/invoke", bytes.NewReader(body)))
	if response.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d: %s", response.Code, response.Body.String())
	}
	if contentType := response.Header().Get("Content-Type"); contentType != "application/json" {
		t.Fatalf("expected JSON error, got %q", contentType)
	}
	if !strings.Contains(response.Body.String(), "plugin_not_installed") {
		t.Fatalf("expected plugin_not_installed code, got %s", response.Body.String())
	}
}

func TestInvokeTimeout(t *testing.T) {
	requireUV(t)
	server := newRuntimeServer(t)

	manifest := "apiVersion: marie.ai/v1alpha1\nkind: ExtensionPackage\nmetadata:\n  id: ext.test.slow\n  author: marie\n  name: slow\n  version: 0.0.1\nruntime:\n  type: python_source\n  language: python\n  version: \"3.12\"\n  entrypoint: main\n"
	script := `import json
import sys
import threading
import time

def heartbeats():
    while True:
        sys.stdout.write(json.dumps({"session_id": "", "event": "heartbeat", "data": None}) + "\n")
        sys.stdout.flush()
        time.sleep(1)

threading.Thread(target=heartbeats, daemon=True).start()
for line in sys.stdin:
    pass
`
	archive := zipArchive(t, map[string][]byte{
		"marie-extension.yaml": []byte(manifest),
		"main.py":              []byte(script),
	})
	install := installArchive(t, server, archive)
	if install.Code != http.StatusOK {
		t.Fatalf("install returned %d: %s", install.Code, install.Body.String())
	}

	envelope := runtimeEnvelope(t, map[string]any{
		"packageRef":    "ext.test.slow",
		"payload":       map[string]any{"ping": true},
		"runtimePolicy": map[string]any{"timeoutMs": float64(300), "maxConcurrent": float64(1), "maxMemoryBytes": float64(536870912), "networkPolicy": "none"},
	})
	body, err := json.Marshal(envelope)
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/dispatch/invoke", bytes.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("expected SSE 200, got %d: %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), `"type":"error"`) || !strings.Contains(response.Body.String(), "timeout") {
		t.Fatalf("expected timeout error frame, got %s", response.Body.String())
	}
}

const versionScriptTemplate = `import json
import sys
import threading
import time

def heartbeats():
    while True:
        sys.stdout.write(json.dumps({"session_id": "", "event": "heartbeat", "data": None}) + "\n")
        sys.stdout.flush()
        time.sleep(1)

threading.Thread(target=heartbeats, daemon=True).start()
for line in sys.stdin:
    request = json.loads(line)
    sid = request.get("session_id")
    if not sid:
        continue
    sys.stdout.write(json.dumps({"session_id": sid, "event": "session", "data": {"type": "stream", "data": {"v": %d}}}) + "\n")
    sys.stdout.write(json.dumps({"session_id": sid, "event": "session", "data": {"type": "end", "data": {}}}) + "\n")
    sys.stdout.flush()
`

const slowStartScript = `import json
import sys
import threading
import time

time.sleep(2)

def heartbeats():
    while True:
        sys.stdout.write(json.dumps({"session_id": "", "event": "heartbeat", "data": None}) + "\n")
        sys.stdout.flush()
        time.sleep(1)

threading.Thread(target=heartbeats, daemon=True).start()
for line in sys.stdin:
    pass
`

func pluginArchive(t *testing.T, packageRef, script string) []byte {
	t.Helper()
	manifest := "apiVersion: marie.ai/v1alpha1\nkind: ExtensionPackage\nmetadata:\n  id: " + packageRef + "\n  author: marie\n  name: test\n  version: 0.0.1\nruntime:\n  type: python_source\n  language: python\n  version: \"3.12\"\n  entrypoint: main\n"
	return zipArchive(t, map[string][]byte{
		"marie-extension.yaml": []byte(manifest),
		"main.py":              []byte(script),
	})
}

func invokeRef(t *testing.T, server http.Handler, packageRef string, payload any) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(runtimeEnvelope(t, map[string]any{"packageRef": packageRef, "payload": payload}))
	if err != nil {
		t.Fatal(err)
	}
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/dispatch/invoke", bytes.NewReader(body)))
	return response
}

func TestReinstallReplacesRunningInstance(t *testing.T) {
	requireUV(t)
	server := newRuntimeServer(t)

	v1 := pluginArchive(t, "ext.test.version", fmt.Sprintf(versionScriptTemplate, 1))
	if response := installArchive(t, server, v1); response.Code != http.StatusOK {
		t.Fatalf("install v1 returned %d: %s", response.Code, response.Body.String())
	}
	invoke := invokeRef(t, server, "ext.test.version", map[string]any{})
	if invoke.Code != http.StatusOK || !strings.Contains(invoke.Body.String(), `"v":1`) {
		t.Fatalf("expected v1 response, got %d: %s", invoke.Code, invoke.Body.String())
	}

	v2 := pluginArchive(t, "ext.test.version", fmt.Sprintf(versionScriptTemplate, 2))
	if response := installArchive(t, server, v2); response.Code != http.StatusOK {
		t.Fatalf("install v2 returned %d: %s", response.Code, response.Body.String())
	}
	invoke = invokeRef(t, server, "ext.test.version", map[string]any{})
	if invoke.Code != http.StatusOK || !strings.Contains(invoke.Body.String(), `"v":2`) {
		t.Fatalf("expected v2 response after re-install, got %d: %s", invoke.Code, invoke.Body.String())
	}
}

func TestHealthRespondsDuringDeploy(t *testing.T) {
	requireUV(t)
	server := newRuntimeServer(t)

	archive := pluginArchive(t, "ext.test.slowstart", slowStartScript)
	header := envelopeHeader(t, installEnvelope(t, archive, nil))
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() {
		request := httptest.NewRequest(http.MethodPost, "/v1/plugins/install", bytes.NewReader(archive))
		request.Header.Set("X-Marie-Envelope", header)
		response := httptest.NewRecorder()
		server.ServeHTTP(response, request)
		done <- response
	}()

	// Let the deploy get in flight (the plugin sleeps 2s before its first
	// heartbeat), then health must answer without waiting for it.
	time.Sleep(500 * time.Millisecond)
	started := time.Now()
	health := httptest.NewRecorder()
	server.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/health", nil))
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("health blocked behind in-flight deploy: %s", elapsed)
	}
	if health.Code != http.StatusOK {
		t.Fatalf("health returned %d", health.Code)
	}

	select {
	case response := <-done:
		if response.Code != http.StatusOK {
			t.Fatalf("install returned %d: %s", response.Code, response.Body.String())
		}
	case <-time.After(30 * time.Second):
		t.Fatal("install never finished")
	}
}

func TestInstallRejectsClaimMismatch(t *testing.T) {
	server := newRuntimeServer(t)
	archive := fixtureArchive(t)

	request := httptest.NewRequest(http.MethodPost, "/v1/plugins/install", bytes.NewReader(archive))
	request.Header.Set("X-Marie-Envelope", envelopeHeader(t, installEnvelope(t, archive, map[string]any{"packageDigest": "sha256:wrong"})))
	response := httptest.NewRecorder()
	server.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "claim_mismatch") {
		t.Fatalf("expected 400 claim_mismatch, got %d: %s", response.Code, response.Body.String())
	}

	list := httptest.NewRecorder()
	listRequest := httptest.NewRequest(http.MethodGet, "/v1/plugins", nil)
	listRequest.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, nil)))
	server.ServeHTTP(list, listRequest)
	if !strings.Contains(list.Body.String(), `"plugins":[]`) {
		t.Fatalf("mismatched install must be rolled back, got %s", list.Body.String())
	}

	remove := httptest.NewRecorder()
	removeRequest := httptest.NewRequest(http.MethodDelete, "/v1/plugins/ext.test.other", nil)
	removeRequest.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, nil)))
	server.ServeHTTP(remove, removeRequest)
	if remove.Code != http.StatusBadRequest || !strings.Contains(remove.Body.String(), "claim_mismatch") {
		t.Fatalf("expected 400 claim_mismatch on delete, got %d: %s", remove.Code, remove.Body.String())
	}
}

func TestInstallClaimMismatchPreservesExistingInstall(t *testing.T) {
	requireUV(t)
	server := newRuntimeServer(t)

	if response := installArchive(t, server, fixtureArchive(t)); response.Code != http.StatusOK {
		t.Fatalf("install returned %d: %s", response.Code, response.Body.String())
	}

	// Same packageRef, different content, digest claim matching neither.
	imposter := pluginArchive(t, fixturePackageRef, fmt.Sprintf(versionScriptTemplate, 9))
	request := httptest.NewRequest(http.MethodPost, "/v1/plugins/install", bytes.NewReader(imposter))
	request.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, map[string]any{"packageDigest": "sha256:wrong"})))
	response := httptest.NewRecorder()
	server.ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "claim_mismatch") {
		t.Fatalf("expected 400 claim_mismatch, got %d: %s", response.Code, response.Body.String())
	}

	list := httptest.NewRecorder()
	listRequest := httptest.NewRequest(http.MethodGet, "/v1/plugins", nil)
	listRequest.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, nil)))
	server.ServeHTTP(list, listRequest)
	if !strings.Contains(list.Body.String(), fixturePackageRef) || !strings.Contains(list.Body.String(), `"state":"ready"`) {
		t.Fatalf("existing install must survive a mismatched upload, got %s", list.Body.String())
	}

	invoke := invokeRef(t, server, fixturePackageRef, map[string]any{"ping": true})
	if invoke.Code != http.StatusOK || !strings.Contains(invoke.Body.String(), `"echo"`) || !strings.Contains(invoke.Body.String(), `"type":"end"`) {
		t.Fatalf("existing install must remain invokable, got %d: %s", invoke.Code, invoke.Body.String())
	}
}

func TestInstallOversizedEntryRejected(t *testing.T) {
	server := newRuntimeServer(t)

	manifest, err := os.ReadFile(filepath.Join("..", "..", "testdata", "fixture-plugin", "marie-extension.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	archive := zipArchive(t, map[string][]byte{
		"marie-extension.yaml": manifest,
		"big.bin":              make([]byte, (256<<20)+1),
	})

	// Claims are irrelevant: decoding fails before they are compared.
	request := httptest.NewRequest(http.MethodPost, "/v1/plugins/install", bytes.NewReader(archive))
	request.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, nil)))
	response := httptest.NewRecorder()
	server.ServeHTTP(response, request)
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d: %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "archive_too_large") {
		t.Fatalf("expected archive_too_large code, got %s", response.Body.String())
	}
}

func TestTenantIsolation(t *testing.T) {
	requireUV(t)
	server := newRuntimeServer(t)

	if response := installArchive(t, server, fixtureArchive(t)); response.Code != http.StatusOK {
		t.Fatalf("install returned %d: %s", response.Code, response.Body.String())
	}

	list := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/plugins", nil)
	request.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, map[string]any{"organizationId": "org2"})))
	server.ServeHTTP(list, request)
	if list.Code != http.StatusOK || !strings.Contains(list.Body.String(), `"plugins":[]`) {
		t.Fatalf("expected empty list for other tenant, got %d: %s", list.Code, list.Body.String())
	}
}

func TestRejectsAmbiguousTenantClaims(t *testing.T) {
	server := newRuntimeServer(t)

	list := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/v1/plugins", nil)
	request.Header.Set("X-Marie-Envelope", envelopeHeader(t, runtimeEnvelope(t, map[string]any{"organizationId": "org__1"})))
	server.ServeHTTP(list, request)
	if list.Code != http.StatusBadRequest || !strings.Contains(list.Body.String(), "invalid_claims") {
		t.Fatalf("expected 400 invalid_claims, got %d: %s", list.Code, list.Body.String())
	}
}

func TestInvokeOversizedBodyRejected(t *testing.T) {
	server := newRuntimeServer(t)

	body := append([]byte(`{"x":"`), bytes.Repeat([]byte("a"), 5<<20)...)
	body = append(body, []byte(`"}`)...)
	response := httptest.NewRecorder()
	server.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/dispatch/invoke", bytes.NewReader(body)))
	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d: %s", response.Code, response.Body.String())
	}
	if !strings.Contains(response.Body.String(), "body_too_large") {
		t.Fatalf("expected body_too_large code, got %s", response.Body.String())
	}
}

func TestHealthCounts(t *testing.T) {
	requireUV(t)
	server := newRuntimeServer(t)

	if code := installArchive(t, server, fixtureArchive(t)).Code; code != http.StatusOK {
		t.Fatalf("install returned %d", code)
	}

	health := httptest.NewRecorder()
	server.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/health", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("health returned %d", health.Code)
	}
	parsed := struct {
		Plugins   int `json:"plugins"`
		Instances int `json:"instances"`
	}{}
	if err := json.Unmarshal(health.Body.Bytes(), &parsed); err != nil {
		t.Fatal(err)
	}
	if parsed.Plugins < 1 || parsed.Instances < 1 {
		t.Fatalf("expected counts >= 1, got %+v", parsed)
	}
}
