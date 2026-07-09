package httpapi

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/plugin_manager"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/auth"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/decoder"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/policy"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/service"
)

const (
	maxInstallArchiveBytes = 256 << 20
	maxJSONBodyBytes       = 4 << 20
	maxInvokeTimeout       = 10 * time.Minute
)

type VersionInfo struct {
	Version string `json:"version"`
	Commit  string `json:"commit"`
	Mode    string `json:"mode"`
}

type decodeRequest struct {
	Path string `json:"path"`
}

type serverConfig struct {
	verifier *auth.EnvelopeVerifier
	manager  *plugin_manager.Manager
	pool     *io_tunnel.Pool
}

type ServerOption func(*serverConfig)

func WithEnvelopeVerifier(verifier *auth.EnvelopeVerifier) ServerOption {
	return func(config *serverConfig) {
		config.verifier = verifier
	}
}

func WithManager(manager *plugin_manager.Manager) ServerOption {
	return func(config *serverConfig) {
		config.manager = manager
	}
}

func WithPool(pool *io_tunnel.Pool) ServerOption {
	return func(config *serverConfig) {
		config.pool = pool
	}
}

func NewServer(version VersionInfo, options ...ServerOption) http.Handler {
	config := serverConfig{}
	for _, option := range options {
		option(&config)
	}

	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		pluginCount := 0
		instanceCount := 0
		if config.manager != nil {
			pluginCount = config.manager.Count()
		}
		if config.pool != nil {
			instanceCount = config.pool.ReadyCount()
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"ok":        true,
			"ready":     true,
			"version":   version.Version,
			"mode":      version.Mode,
			"plugins":   pluginCount,
			"instances": instanceCount,
		})
	})

	mux.HandleFunc("GET /version", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, version)
	})

	mux.HandleFunc("POST /v1/packages/decode", func(w http.ResponseWriter, r *http.Request) {
		var input decodeRequest
		if !decodeJSONBody(w, r, &input) {
			return
		}
		if input.Path == "" {
			writeJSON(w, http.StatusBadRequest, errorBody("missing_path", "path is required"))
			return
		}

		result, err := decoder.DecodePath(input.Path)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, errorBody("decode_failed", err.Error()))
			return
		}
		writeJSON(w, http.StatusOK, result)
	})

	mux.HandleFunc("POST /v1/runtime/stub-invocations", func(w http.ResponseWriter, r *http.Request) {
		var envelope map[string]any
		if !decodeJSONBody(w, r, &envelope) {
			return
		}
		if err := config.verifier.Verify(envelope); err != nil {
			writeJSON(w, http.StatusUnauthorized, errorBody(auth.Code(err), err.Error()))
			return
		}
		if err := policy.VerifyRuntimeEnvelope(envelope); err != nil {
			writeJSON(w, http.StatusForbidden, errorBody(policy.Code(err), err.Error()))
			return
		}
		writeJSON(w, http.StatusNotImplemented, errorBody("runtime_disabled", "runtime invocation is disabled in decode-only mode"))
	})

	mux.HandleFunc("POST /v1/plugins/install", func(w http.ResponseWriter, r *http.Request) {
		envelope, tenant, ok := authorizeHeaderEnvelope(w, r, config)
		if !ok {
			return
		}

		r.Body = http.MaxBytesReader(w, r.Body, maxInstallArchiveBytes)
		archive, err := io.ReadAll(r.Body)
		if err != nil {
			maxBytesErr := &http.MaxBytesError{}
			if errors.As(err, &maxBytesErr) {
				writeJSON(w, http.StatusRequestEntityTooLarge, errorBody("archive_too_large", err.Error()))
				return
			}
			writeJSON(w, http.StatusBadRequest, errorBody("invalid_archive", err.Error()))
			return
		}

		// Verify the signed claims against the archive before the install
		// replaces any existing version.
		packageRef, digest, err := plugin_manager.Inspect(archive)
		if err != nil {
			writeInstallError(w, err)
			return
		}
		if stringClaim(envelope["packageRef"]) != packageRef || stringClaim(envelope["packageDigest"]) != digest {
			writeJSON(w, http.StatusBadRequest, errorBody("claim_mismatch",
				"envelope packageRef/packageDigest do not match the uploaded archive"))
			return
		}

		install, err := config.manager.Install(tenant, archive)
		if err != nil {
			writeInstallError(w, err)
			return
		}
		install, err = config.pool.Deploy(r.Context(), tenant, install.PackageRef)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, errorBody("deploy_failed", err.Error()))
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"install": install, "state": install.State})
	})

	mux.HandleFunc("GET /v1/plugins", func(w http.ResponseWriter, r *http.Request) {
		_, tenant, ok := authorizeHeaderEnvelope(w, r, config)
		if !ok {
			return
		}

		installs := config.manager.List(tenant)
		for index := range installs {
			if state := config.pool.InstanceState(tenant, installs[index].PackageRef); state != io_tunnel.StateAbsent {
				installs[index].State = string(state)
			}
		}
		writeJSON(w, http.StatusOK, map[string]any{"plugins": installs})
	})

	mux.HandleFunc("DELETE /v1/plugins/{packageRef}", func(w http.ResponseWriter, r *http.Request) {
		envelope, tenant, ok := authorizeHeaderEnvelope(w, r, config)
		if !ok {
			return
		}

		packageRef := r.PathValue("packageRef")
		if stringClaim(envelope["packageRef"]) != packageRef {
			writeJSON(w, http.StatusBadRequest, errorBody("claim_mismatch",
				"envelope packageRef does not match the requested plugin"))
			return
		}
		if err := config.pool.Stop(tenant, packageRef); err != nil {
			writeJSON(w, http.StatusInternalServerError, errorBody("stop_failed", err.Error()))
			return
		}
		if err := config.manager.Remove(tenant, packageRef); err != nil {
			if errors.Is(err, plugin_manager.ErrInstallNotFound) {
				writeJSON(w, http.StatusNotFound, errorBody("install_not_found", err.Error()))
				return
			}
			writeJSON(w, http.StatusBadRequest, errorBody("remove_failed", err.Error()))
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"removed": packageRef})
	})

	mux.HandleFunc("POST /v1/dispatch/invoke", func(w http.ResponseWriter, r *http.Request) {
		if !runtimeConfigured(w, config) {
			return
		}
		var envelope map[string]any
		if !decodeJSONBody(w, r, &envelope) {
			return
		}
		tenant, ok := authorizeEnvelope(w, config, envelope)
		if !ok {
			return
		}

		packageRef := stringClaim(envelope["packageRef"])
		if _, installed := config.manager.Get(tenant, packageRef); !installed {
			writeJSON(w, http.StatusNotFound, errorBody("plugin_not_installed", "plugin is not installed: "+packageRef))
			return
		}

		frames, err := config.pool.Invoke(r.Context(), tenant, packageRef, envelope["payload"], envelopeTimeout(envelope))
		if errors.Is(err, io_tunnel.ErrNotRunning) {
			// The plugin is installed but has no running instance (e.g. after a
			// daemon restart). Lazy-deploy it from its existing working dir and
			// retry once — Deploy needs only the packageRef, not the archive.
			if _, derr := config.pool.Deploy(r.Context(), tenant, packageRef); derr != nil {
				writeJSON(w, http.StatusInternalServerError, errorBody("deploy_failed", derr.Error()))
				return
			}
			frames, err = config.pool.Invoke(r.Context(), tenant, packageRef, envelope["payload"], envelopeTimeout(envelope))
		}
		if err != nil {
			if errors.Is(err, io_tunnel.ErrNotRunning) {
				writeJSON(w, http.StatusConflict, errorBody("instance_not_running", err.Error()))
				return
			}
			writeJSON(w, http.StatusInternalServerError, errorBody("invoke_failed", err.Error()))
			return
		}

		sse := service.NewSSEWriter(w)
		for frame := range frames {
			if err := sse.WriteData(frame); err != nil {
				// Client is gone; returning cancels r.Context() which unwinds
				// the pool consumer.
				return
			}
		}
	})

	return mux
}

func runtimeConfigured(w http.ResponseWriter, config serverConfig) bool {
	if config.manager == nil || config.pool == nil {
		writeJSON(w, http.StatusServiceUnavailable, errorBody("runtime_unconfigured", "plugin runtime is not configured"))
		return false
	}
	return true
}

func authorizeHeaderEnvelope(w http.ResponseWriter, r *http.Request, config serverConfig) (map[string]any, string, bool) {
	if !runtimeConfigured(w, config) {
		return nil, "", false
	}
	raw := r.Header.Get("X-Marie-Envelope")
	if raw == "" {
		writeJSON(w, http.StatusUnauthorized, errorBody("missing_envelope", "X-Marie-Envelope header is required"))
		return nil, "", false
	}
	var envelope map[string]any
	if err := json.Unmarshal([]byte(raw), &envelope); err != nil {
		writeJSON(w, http.StatusUnauthorized, errorBody("invalid_envelope", err.Error()))
		return nil, "", false
	}
	tenant, ok := authorizeEnvelope(w, config, envelope)
	return envelope, tenant, ok
}

func authorizeEnvelope(w http.ResponseWriter, config serverConfig, envelope map[string]any) (string, bool) {
	if err := config.verifier.Verify(envelope); err != nil {
		writeJSON(w, http.StatusUnauthorized, errorBody(auth.Code(err), err.Error()))
		return "", false
	}
	if err := policy.VerifyRuntimeEnvelope(envelope); err != nil {
		writeJSON(w, http.StatusForbidden, errorBody(policy.Code(err), err.Error()))
		return "", false
	}

	organization := strings.TrimSpace(stringClaim(envelope["organizationId"]))
	workspace := strings.TrimSpace(stringClaim(envelope["workspaceId"]))
	if organization == "" || workspace == "" {
		writeJSON(w, http.StatusForbidden, errorBody("tenant_claims_required", "organizationId and workspaceId claims are required"))
		return "", false
	}
	if strings.Contains(organization, "__") || strings.Contains(workspace, "__") {
		writeJSON(w, http.StatusBadRequest, errorBody("invalid_claims", "organizationId and workspaceId must not contain __"))
		return "", false
	}
	return organization + "__" + workspace, true
}

func decodeJSONBody(w http.ResponseWriter, r *http.Request, into any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, maxJSONBodyBytes)
	if err := json.NewDecoder(r.Body).Decode(into); err != nil {
		maxBytesErr := &http.MaxBytesError{}
		if errors.As(err, &maxBytesErr) {
			writeJSON(w, http.StatusRequestEntityTooLarge, errorBody("body_too_large", err.Error()))
			return false
		}
		writeJSON(w, http.StatusBadRequest, errorBody("invalid_json", err.Error()))
		return false
	}
	return true
}

func envelopeTimeout(envelope map[string]any) time.Duration {
	runtimePolicy, _ := envelope["runtimePolicy"].(map[string]any)
	timeoutMs, _ := runtimePolicy["timeoutMs"].(float64)
	timeout := time.Duration(timeoutMs) * time.Millisecond
	if timeout > maxInvokeTimeout {
		timeout = maxInvokeTimeout
	}
	return timeout
}

func writeInstallError(w http.ResponseWriter, err error) {
	code := installErrorCode(err)
	status := http.StatusBadRequest
	if code == "archive_too_large" {
		status = http.StatusRequestEntityTooLarge
	}
	writeJSON(w, status, errorBody(code, err.Error()))
}

func installErrorCode(err error) string {
	switch {
	case errors.Is(err, plugin_manager.ErrInvalidName):
		return "invalid_name"
	case errors.Is(err, plugin_manager.ErrInvalidPackage):
		return "invalid_package"
	case errors.Is(err, plugin_manager.ErrUnsafeArchivePath):
		return "unsafe_archive_path"
	case errors.Is(err, plugin_manager.ErrArchiveTooLarge), errors.Is(err, decoder.ErrPackageTooLarge):
		return "archive_too_large"
	default:
		return "decode_failed"
	}
}

func stringClaim(value any) string {
	typed, _ := value.(string)
	return typed
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func errorBody(code string, message string) map[string]any {
	return map[string]any{
		"error": map[string]string{
			"code":    code,
			"message": message,
		},
	}
}
