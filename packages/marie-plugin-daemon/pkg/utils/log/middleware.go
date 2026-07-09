package log

import (
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"
)

// Middleware is a standard net/http middleware.
type Middleware func(http.Handler) http.Handler

// Chain composes middlewares around a handler (first listed runs outermost).
func Chain(h http.Handler, mws ...Middleware) http.Handler {
	for i := len(mws) - 1; i >= 0; i-- {
		h = mws[i](h)
	}
	return h
}

// TraceMiddleware ensures every request carries a trace (from the inbound
// `traceparent` header or freshly generated) plus any identity headers, so all
// logs emitted during the request are correlated.
func TraceMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		traceID, spanID, ok := ParseTraceparent(r.Header.Get("traceparent"))
		if !ok {
			traceID = GenerateTraceID()
			spanID = GenerateSpanID()
		}
		ctx := WithTrace(r.Context(), TraceContext{TraceID: traceID, SpanID: spanID})

		identity := Identity{
			UserID:   r.Header.Get("X-User-ID"),
			UserType: r.Header.Get("X-User-Type"),
		}
		if identity.UserID != "" || identity.UserType != "" {
			ctx = WithIdentity(ctx, identity)
		}
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// LoggerMiddleware logs one line per request (method/path/status/latency/ip),
// at a level chosen by status code. Paths in skip are not logged (e.g. /health).
func LoggerMiddleware(skip ...string) Middleware {
	skipped := make(map[string]bool, len(skip))
	for _, p := range skip {
		skipped[p] = true
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if skipped[r.URL.Path] {
				next.ServeHTTP(w, r)
				return
			}
			start := time.Now()
			rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
			next.ServeHTTP(rec, r)

			path := r.URL.Path
			if r.URL.RawQuery != "" {
				path += "?" + r.URL.RawQuery
			}
			level := slog.LevelInfo
			switch {
			case rec.status >= 500:
				level = slog.LevelError
			case rec.status >= 400:
				level = slog.LevelWarn
			}
			slog.Log(r.Context(), level, "HTTP request",
				"method", r.Method,
				"path", path,
				"status", rec.status,
				"latency_ms", time.Since(start).Milliseconds(),
				"client_ip", clientIP(r),
			)
		})
	}
}

// RecoveryMiddleware turns a handler panic into a logged stack trace + 500.
func RecoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				slog.ErrorContext(r.Context(), "panic recovered",
					"error", fmt.Sprintf("%v", err),
					"stack_trace", captureFullPanicStack(),
				)
				w.WriteHeader(http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// statusRecorder captures the response status while forwarding Flush so that
// streaming (SSE) responses are not buffered by the middleware wrapper.
type statusRecorder struct {
	http.ResponseWriter
	status  int
	written bool
}

func (r *statusRecorder) WriteHeader(code int) {
	if !r.written {
		r.status = code
		r.written = true
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	r.written = true
	return r.ResponseWriter.Write(b)
}

func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		return xff
	}
	if host, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return host
	}
	return r.RemoteAddr
}
