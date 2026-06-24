// Package log is the marie-plugin-daemon's structured logging façade.
//
// Ported from the upstream dify-plugin-daemon logger: a slog-based logger with
// levels, JSON/text output, optional file output, caller info, trace/identity
// context, automatic stack traces on errors, and panic recovery. No external
// dependencies (standard library only).
package log

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

const ServiceName = "marie-plugin-daemon"

// ParseLevel maps a string log level to a slog.Level.
func ParseLevel(value string) (slog.Level, error) {
	switch value {
	case "":
		return slog.LevelInfo, nil
	case "DEBUG":
		return slog.LevelDebug, nil
	case "INFO":
		return slog.LevelInfo, nil
	case "WARN":
		return slog.LevelWarn, nil
	case "ERROR":
		return slog.LevelError, nil
	default:
		return 0, fmt.Errorf("invalid log level %q. Valid values are: DEBUG, INFO, WARN, ERROR", value)
	}
}

// Init configures the default logger. When filename is non-empty, output is
// teed to stdout and that file. Returns a closer for the file (nil if none).
func Init(json bool, filename string, level string) (io.Closer, error) {
	var w io.Writer = os.Stdout
	var closer io.Closer
	if filename != "" {
		dir := filepath.Dir(filename)
		if err := os.MkdirAll(dir, 0700); err != nil {
			return nil, fmt.Errorf("create log directory %q: %w", dir, err)
		}
		file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
		if err != nil {
			return nil, fmt.Errorf("open log file %q: %w", filename, err)
		}
		w = io.MultiWriter(os.Stdout, file)
		closer = file
	}

	logLevel, err := ParseLevel(level)
	if err != nil {
		if closer != nil {
			_ = closer.Close()
		}
		return nil, err
	}

	handler := NewHandler(Options{
		Level:   logLevel,
		Service: ServiceName,
		JSON:    json,
		Out:     w,
	})
	slog.SetDefault(slog.New(handler))
	return closer, nil
}

func logWithCaller(ctx context.Context, level slog.Level, msg string, args ...any) {
	logger := slog.Default()
	if !logger.Enabled(ctx, level) {
		return
	}
	var pcs [1]uintptr
	runtime.Callers(3, pcs[:])
	r := slog.NewRecord(time.Now(), level, msg, pcs[0])
	r.Add(args...)
	_ = logger.Handler().Handle(ctx, r)
}

func Debug(msg string, args ...any) {
	logWithCaller(context.Background(), slog.LevelDebug, msg, args...)
}

func Info(msg string, args ...any) {
	logWithCaller(context.Background(), slog.LevelInfo, msg, args...)
}

func Warn(msg string, args ...any) {
	logWithCaller(context.Background(), slog.LevelWarn, msg, args...)
}

func Error(msg string, args ...any) {
	logWithCaller(context.Background(), slog.LevelError, msg, args...)
}

func Panic(msg string, args ...any) {
	logWithCaller(context.Background(), slog.LevelError, msg, args...)
	panic(msg)
}

func DebugContext(ctx context.Context, msg string, args ...any) {
	logWithCaller(ctx, slog.LevelDebug, msg, args...)
}

func InfoContext(ctx context.Context, msg string, args ...any) {
	logWithCaller(ctx, slog.LevelInfo, msg, args...)
}

func WarnContext(ctx context.Context, msg string, args ...any) {
	logWithCaller(ctx, slog.LevelWarn, msg, args...)
}

func ErrorContext(ctx context.Context, msg string, args ...any) {
	logWithCaller(ctx, slog.LevelError, msg, args...)
}

func PanicContext(ctx context.Context, msg string, args ...any) {
	logWithCaller(ctx, slog.LevelError, msg, args...)
	panic(msg)
}

// RecoverAndExit logs a recovered panic with its stack and exits non-zero.
func RecoverAndExit() {
	if err := recover(); err != nil {
		stack := captureFullPanicStack()
		slog.Error("panic recovered",
			"error", fmt.Sprintf("%v", err),
			"stack_trace", stack,
		)
		os.Exit(1)
	}
}

func captureFullPanicStack() string {
	buf := make([]byte, 4096)
	for {
		n := runtime.Stack(buf, false)
		if n < len(buf) {
			return string(buf[:n])
		}
		buf = make([]byte, len(buf)*2)
	}
}
