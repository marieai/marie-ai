package auth

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestVerifySignedEnvelope(t *testing.T) {
	now := time.Date(2026, 6, 2, 12, 0, 0, 0, time.UTC)
	verifier := NewEnvelopeVerifier([]SigningKey{{KeyID: "marie-api-test", Secret: "test-runtime-secret"}}, func() time.Time {
		return now
	})

	envelope := signedEnvelope(t, verifier, "nonce-1", now.Add(5*time.Minute))
	if err := verifier.Verify(envelope); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyStudioSignedFixture(t *testing.T) {
	now := time.Date(2026, 6, 2, 12, 0, 0, 0, time.UTC)
	verifier := NewEnvelopeVerifier([]SigningKey{{KeyID: "marie-api-test", Secret: "test-runtime-secret"}}, func() time.Time {
		return now
	})

	data, err := os.ReadFile(filepath.Join(testdataRoot(t), "studio-signed-envelope.json"))
	if err != nil {
		t.Fatal(err)
	}
	envelope := map[string]any{}
	if err := json.Unmarshal(data, &envelope); err != nil {
		t.Fatal(err)
	}

	if err := verifier.Verify(envelope); err != nil {
		t.Fatal(err)
	}
}

func TestRejectMissingSignature(t *testing.T) {
	now := time.Date(2026, 6, 2, 12, 0, 0, 0, time.UTC)
	verifier := NewEnvelopeVerifier([]SigningKey{{KeyID: "marie-api-test", Secret: "test-runtime-secret"}}, func() time.Time {
		return now
	})
	envelope := unsignedEnvelope("nonce-1", now.Add(5*time.Minute))

	if err := verifier.Verify(envelope); Code(err) != "missing_signature" {
		t.Fatalf("expected missing signature, got %v", err)
	}
}

func TestRejectExpiredEnvelope(t *testing.T) {
	now := time.Date(2026, 6, 2, 12, 0, 0, 0, time.UTC)
	verifier := NewEnvelopeVerifier([]SigningKey{{KeyID: "marie-api-test", Secret: "test-runtime-secret"}}, func() time.Time {
		return now
	})
	envelope := signedEnvelope(t, verifier, "nonce-1", now.Add(-time.Second))

	if err := verifier.Verify(envelope); Code(err) != "expired_envelope" {
		t.Fatalf("expected expired envelope, got %v", err)
	}
}

func TestRejectReplayNonce(t *testing.T) {
	now := time.Date(2026, 6, 2, 12, 0, 0, 0, time.UTC)
	verifier := NewEnvelopeVerifier([]SigningKey{{KeyID: "marie-api-test", Secret: "test-runtime-secret"}}, func() time.Time {
		return now
	})
	envelope := signedEnvelope(t, verifier, "nonce-1", now.Add(5*time.Minute))

	if err := verifier.Verify(envelope); err != nil {
		t.Fatal(err)
	}
	replay := signedEnvelope(t, verifier, "nonce-1", now.Add(5*time.Minute))
	if err := verifier.Verify(replay); Code(err) != "replay_nonce" {
		t.Fatalf("expected replay nonce, got %v", err)
	}
}

func TestRejectInvalidRuntimePolicy(t *testing.T) {
	now := time.Date(2026, 6, 2, 12, 0, 0, 0, time.UTC)
	verifier := NewEnvelopeVerifier([]SigningKey{{KeyID: "marie-api-test", Secret: "test-runtime-secret"}}, func() time.Time {
		return now
	})
	envelope := signedEnvelope(t, verifier, "nonce-1", now.Add(5*time.Minute))
	envelope["runtimePolicy"] = map[string]any{"timeoutMs": float64(30000), "networkPolicy": "none"}
	envelope["signature"] = map[string]any{
		"keyId":     "marie-api-test",
		"algorithm": "hmac-sha256",
		"value":     signEnvelope(envelope, []byte("test-runtime-secret")),
	}

	if err := verifier.Verify(envelope); Code(err) != "invalid_runtime_policy" {
		t.Fatalf("expected invalid runtime policy, got %v", err)
	}
}

func signedEnvelope(t *testing.T, verifier *EnvelopeVerifier, nonce string, expiresAt time.Time) map[string]any {
	t.Helper()

	envelope := unsignedEnvelope(nonce, expiresAt)
	envelope["signature"] = map[string]any{
		"keyId":     "marie-api-test",
		"algorithm": "hmac-sha256",
		"value":     signEnvelope(envelope, verifier.keys["marie-api-test"]),
	}
	return envelope
}

func unsignedEnvelope(nonce string, expiresAt time.Time) map[string]any {
	return map[string]any{
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
}

func testdataRoot(t *testing.T) string {
	t.Helper()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime caller unavailable")
	}
	return filepath.Join(filepath.Dir(file), "testdata")
}
