package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	ErrMissingSignature = errors.New("missing_signature")
	ErrInvalidSignature = errors.New("invalid_signature")
	ErrExpiredEnvelope  = errors.New("expired_envelope")
	ErrReplayNonce      = errors.New("replay_nonce")
	ErrInvalidPolicy    = errors.New("invalid_runtime_policy")
)

type SigningKey struct {
	KeyID  string
	Secret string
}

type EnvelopeVerifier struct {
	keys map[string][]byte
	now  func() time.Time

	// insecure, when true, makes Verify accept ANY envelope without checking the
	// signature. DEV ONLY — used for local end-to-end testing before envelope
	// signing is wired on the marie-ai side.
	insecure bool

	mu     sync.Mutex
	nonces map[string]time.Time
}

type signature struct {
	KeyID     string `json:"keyId"`
	Algorithm string `json:"algorithm"`
	Value     string `json:"value"`
}

func NewEnvelopeVerifier(keys []SigningKey, now func() time.Time) *EnvelopeVerifier {
	lookup := map[string][]byte{}
	for _, key := range keys {
		if key.KeyID == "" || key.Secret == "" {
			continue
		}
		lookup[key.KeyID] = []byte(key.Secret)
	}
	if now == nil {
		now = time.Now
	}

	return &EnvelopeVerifier{
		keys:   lookup,
		now:    now,
		nonces: map[string]time.Time{},
	}
}

// NewInsecureEnvelopeVerifier returns a verifier that accepts ANY envelope without
// checking its signature. DEV ONLY — for local end-to-end testing before envelope
// signing is wired on the marie-ai side. Never enable in a shared environment.
// The runtime policy checks (package/action/mode claims) still apply.
func NewInsecureEnvelopeVerifier() *EnvelopeVerifier {
	return &EnvelopeVerifier{
		keys:     map[string][]byte{},
		now:      time.Now,
		insecure: true,
		nonces:   map[string]time.Time{},
	}
}

func (verifier *EnvelopeVerifier) Verify(envelope map[string]any) error {
	if verifier != nil && verifier.insecure {
		return nil
	}
	if verifier == nil || len(verifier.keys) == 0 {
		return fmt.Errorf("%w: no verifier keys configured", ErrInvalidSignature)
	}

	sig, err := signatureFromEnvelope(envelope)
	if err != nil {
		return err
	}
	secret, ok := verifier.keys[sig.KeyID]
	if !ok || sig.Algorithm != "hmac-sha256" {
		return ErrInvalidSignature
	}
	if !hmac.Equal([]byte(sig.Value), []byte(signEnvelope(envelope, secret))) {
		return ErrInvalidSignature
	}

	expiresAt, err := time.Parse(time.RFC3339, stringValue(envelope["expiresAt"]))
	if err != nil || !expiresAt.After(verifier.now()) {
		return ErrExpiredEnvelope
	}
	if err := validateRuntimePolicy(envelope["runtimePolicy"]); err != nil {
		return err
	}

	nonce := stringValue(envelope["nonce"])
	if nonce == "" {
		return fmt.Errorf("%w: nonce is required", ErrReplayNonce)
	}

	verifier.mu.Lock()
	defer verifier.mu.Unlock()
	verifier.pruneExpiredNonces()
	if _, seen := verifier.nonces[nonce]; seen {
		return ErrReplayNonce
	}
	verifier.nonces[nonce] = expiresAt

	return nil
}

func Code(err error) string {
	switch {
	case errors.Is(err, ErrMissingSignature):
		return "missing_signature"
	case errors.Is(err, ErrInvalidSignature):
		return "invalid_signature"
	case errors.Is(err, ErrExpiredEnvelope):
		return "expired_envelope"
	case errors.Is(err, ErrReplayNonce):
		return "replay_nonce"
	case errors.Is(err, ErrInvalidPolicy):
		return "invalid_runtime_policy"
	default:
		return "runtime_auth_failed"
	}
}

func signEnvelope(envelope map[string]any, secret []byte) string {
	mac := hmac.New(sha256.New, secret)
	mac.Write(mustCanonicalEnvelope(envelope))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func mustCanonicalEnvelope(envelope map[string]any) []byte {
	payload, err := json.Marshal(canonicalValue(envelope, true))
	if err != nil {
		panic(err)
	}
	return payload
}

func canonicalValue(value any, omitSignature bool) any {
	switch typed := value.(type) {
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			if omitSignature && key == "signature" {
				continue
			}
			keys = append(keys, key)
		}
		sort.Strings(keys)
		ordered := map[string]any{}
		for _, key := range keys {
			ordered[key] = canonicalValue(typed[key], false)
		}
		return ordered
	case []any:
		items := make([]any, 0, len(typed))
		for _, item := range typed {
			items = append(items, canonicalValue(item, false))
		}
		return items
	default:
		return typed
	}
}

func signatureFromEnvelope(envelope map[string]any) (signature, error) {
	raw, ok := envelope["signature"].(map[string]any)
	if !ok {
		return signature{}, ErrMissingSignature
	}

	sig := signature{
		KeyID:     stringValue(raw["keyId"]),
		Algorithm: stringValue(raw["algorithm"]),
		Value:     stringValue(raw["value"]),
	}
	if sig.KeyID == "" || sig.Algorithm == "" || sig.Value == "" {
		return signature{}, ErrMissingSignature
	}
	return sig, nil
}

func validateRuntimePolicy(value any) error {
	policy, ok := value.(map[string]any)
	if !ok {
		return ErrInvalidPolicy
	}
	if !positiveNumber(policy["timeoutMs"]) || !positiveNumber(policy["maxConcurrent"]) || !positiveNumber(policy["maxMemoryBytes"]) {
		return ErrInvalidPolicy
	}
	if strings.TrimSpace(stringValue(policy["networkPolicy"])) == "" {
		return ErrInvalidPolicy
	}
	return nil
}

func positiveNumber(value any) bool {
	number, ok := value.(float64)
	return ok && number > 0 && math.Trunc(number) == number
}

func stringValue(value any) string {
	typed, _ := value.(string)
	return typed
}

func (verifier *EnvelopeVerifier) pruneExpiredNonces() {
	now := verifier.now()
	for nonce, expiresAt := range verifier.nonces {
		if !expiresAt.After(now) {
			delete(verifier.nonces, nonce)
		}
	}
}
