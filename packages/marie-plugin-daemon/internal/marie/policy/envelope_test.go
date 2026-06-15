package policy

import "testing"

func TestVerifyRuntimeEnvelopeAllowsValidClaims(t *testing.T) {
	if err := VerifyRuntimeEnvelope(validEnvelope()); err != nil {
		t.Fatalf("expected valid envelope, got %v", err)
	}
}

func TestVerifyRuntimeEnvelopeRejectsBlockedTrust(t *testing.T) {
	envelope := validEnvelope()
	envelope["packageTrustLevel"] = "blocked"
	if err := VerifyRuntimeEnvelope(envelope); Code(err) != "trust_policy_denied" {
		t.Fatalf("expected trust denial, got %v", err)
	}
}

func TestVerifyRuntimeEnvelopeRejectsInvalidCapability(t *testing.T) {
	envelope := validEnvelope()
	envelope["actionType"] = "python"
	if err := VerifyRuntimeEnvelope(envelope); Code(err) != "capability_denied" {
		t.Fatalf("expected capability denial, got %v", err)
	}
}

func TestVerifyRuntimeEnvelopeRejectsCredentialWithoutProvider(t *testing.T) {
	envelope := validEnvelope()
	envelope["providerId"] = ""
	envelope["credentialBindingIds"] = []any{"66666666-6666-6666-6666-666666666666"}
	if err := VerifyRuntimeEnvelope(envelope); Code(err) != "credential_policy_denied" {
		t.Fatalf("expected credential denial, got %v", err)
	}
}

func TestVerifyRuntimeEnvelopeRejectsDuplicateCredentials(t *testing.T) {
	envelope := validEnvelope()
	envelope["credentialBindingIds"] = []any{"66666666-6666-6666-6666-666666666666", "66666666-6666-6666-6666-666666666666"}
	if err := VerifyRuntimeEnvelope(envelope); Code(err) != "credential_policy_denied" {
		t.Fatalf("expected credential denial, got %v", err)
	}
}

func TestVerifyRuntimeEnvelopeRejectsUnsupportedNetworkPolicy(t *testing.T) {
	envelope := validEnvelope()
	envelope["runtimePolicy"] = map[string]any{"networkPolicy": "public_internet"}
	if err := VerifyRuntimeEnvelope(envelope); Code(err) != "network_policy_denied" {
		t.Fatalf("expected network denial, got %v", err)
	}
}

func validEnvelope() map[string]any {
	return map[string]any{
		"packageId":            "77777777-7777-7777-7777-777777777777",
		"packageRef":           "ext.langgenius.search",
		"packageDigest":        "sha256:package",
		"packageTrustLevel":    "community",
		"providerId":           "55555555-5555-5555-5555-555555555555",
		"actionId":             "tools/search",
		"actionType":           "stub",
		"credentialBindingIds": []any{},
		"runtimePolicy":        map[string]any{"networkPolicy": "none"},
		"mode":                 "stub",
	}
}
