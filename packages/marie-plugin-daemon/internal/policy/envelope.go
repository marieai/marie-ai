package policy

import (
	"errors"
	"fmt"
	"strings"
)

var (
	ErrTrustPolicy      = errors.New("trust_policy_denied")
	ErrCapabilityPolicy = errors.New("capability_denied")
	ErrCredentialPolicy = errors.New("credential_policy_denied")
	ErrNetworkPolicy    = errors.New("network_policy_denied")
)

func VerifyRuntimeEnvelope(envelope map[string]any) error {
	if strings.TrimSpace(stringValue(envelope["packageId"])) == "" ||
		strings.TrimSpace(stringValue(envelope["packageRef"])) == "" ||
		strings.TrimSpace(stringValue(envelope["packageDigest"])) == "" {
		return fmt.Errorf("%w: package identity claims are required", ErrTrustPolicy)
	}
	if stringValue(envelope["packageTrustLevel"]) == "blocked" {
		return fmt.Errorf("%w: package trust is blocked", ErrTrustPolicy)
	}

	actionType := stringValue(envelope["actionType"])
	if !allowedActionType(actionType) || strings.TrimSpace(stringValue(envelope["actionId"])) == "" {
		return fmt.Errorf("%w: action claims are invalid", ErrCapabilityPolicy)
	}
	if stringValue(envelope["mode"]) != "stub" {
		return fmt.Errorf("%w: only stub mode is enabled", ErrCapabilityPolicy)
	}

	credentialIDs, ok := envelope["credentialBindingIds"].([]any)
	if !ok {
		return fmt.Errorf("%w: credential binding ids must be an array", ErrCredentialPolicy)
	}
	if len(credentialIDs) > 0 && strings.TrimSpace(stringValue(envelope["providerId"])) == "" {
		return fmt.Errorf("%w: credential bindings require a provider", ErrCredentialPolicy)
	}
	seen := map[string]struct{}{}
	for _, raw := range credentialIDs {
		id := strings.TrimSpace(stringValue(raw))
		if id == "" {
			return fmt.Errorf("%w: credential binding id is empty", ErrCredentialPolicy)
		}
		if _, exists := seen[id]; exists {
			return fmt.Errorf("%w: duplicate credential binding id", ErrCredentialPolicy)
		}
		seen[id] = struct{}{}
	}

	policy, ok := envelope["runtimePolicy"].(map[string]any)
	if !ok || !allowedNetworkPolicy(stringValue(policy["networkPolicy"])) {
		return fmt.Errorf("%w: runtime network policy is not allowed", ErrNetworkPolicy)
	}
	return nil
}

func Code(err error) string {
	switch {
	case errors.Is(err, ErrTrustPolicy):
		return "trust_policy_denied"
	case errors.Is(err, ErrCapabilityPolicy):
		return "capability_denied"
	case errors.Is(err, ErrCredentialPolicy):
		return "credential_policy_denied"
	case errors.Is(err, ErrNetworkPolicy):
		return "network_policy_denied"
	default:
		return "runtime_policy_denied"
	}
}

func allowedActionType(value string) bool {
	switch value {
	case "tool", "model", "datasource", "trigger", "endpoint", "mcp", "webapp", "stub":
		return true
	default:
		return false
	}
}

func allowedNetworkPolicy(value string) bool {
	switch value {
	case "none", "manifest_declared", "internal_only":
		return true
	default:
		return false
	}
}

func stringValue(value any) string {
	typed, _ := value.(string)
	return typed
}
