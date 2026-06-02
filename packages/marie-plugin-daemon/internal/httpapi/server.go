package httpapi

import (
	"encoding/json"
	"net/http"

	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/decoder"
)

type VersionInfo struct {
	Version string `json:"version"`
	Commit  string `json:"commit"`
	Mode    string `json:"mode"`
}

type decodeRequest struct {
	Path string `json:"path"`
}

func NewServer(version VersionInfo) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"ok":      true,
			"ready":   true,
			"version": version.Version,
			"mode":    version.Mode,
		})
	})

	mux.HandleFunc("GET /version", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, version)
	})

	mux.HandleFunc("POST /v1/packages/decode", func(w http.ResponseWriter, r *http.Request) {
		var input decodeRequest
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			writeJSON(w, http.StatusBadRequest, errorBody("invalid_json", err.Error()))
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

	mux.HandleFunc("POST /v1/runtime/stub-invocations", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusNotImplemented, errorBody("runtime_disabled", "runtime invocation is disabled in decode-only mode"))
	})

	return mux
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
