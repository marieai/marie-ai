package local_runtime

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestEnsureEnvironmentCreatesVenv(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "requirements.txt"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	python, err := EnsureEnvironment(context.Background(), dir, "3.12", io.Discard)
	if err != nil {
		t.Fatalf("environment setup failed: %v", err)
	}
	if _, err := os.Stat(python); err != nil {
		t.Fatalf("venv python missing: %v", err)
	}
}

func TestEnsureEnvironmentIdempotent(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	dir := t.TempDir()
	first, err := EnsureEnvironment(context.Background(), dir, "3.12", io.Discard)
	if err != nil {
		t.Fatalf("first setup failed: %v", err)
	}
	second, err := EnsureEnvironment(context.Background(), dir, "3.12", io.Discard)
	if err != nil {
		t.Fatalf("second setup failed: %v", err)
	}
	if first != second {
		t.Fatalf("expected same path, got %s and %s", first, second)
	}
	if _, err := os.Stat(second); err != nil {
		t.Fatalf("venv python missing: %v", err)
	}
}

func TestEnsureEnvironmentCancelledContext(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := EnsureEnvironment(ctx, t.TempDir(), "3.12", io.Discard)
	if !errors.Is(err, ErrEnvironmentSetup) {
		t.Fatalf("expected ErrEnvironmentSetup, got %v", err)
	}
}

func TestEnsureEnvironmentRejectsInvalidVersion(t *testing.T) {
	_, err := EnsureEnvironment(context.Background(), t.TempDir(), "3.12; rm -rf", io.Discard)
	if !errors.Is(err, ErrEnvironmentSetup) {
		t.Fatalf("expected ErrEnvironmentSetup, got %v", err)
	}
}

func TestPythonPathWithoutVenvFallsBackToSystem(t *testing.T) {
	python := PythonPath(t.TempDir())
	if python != "python3" {
		t.Fatalf("expected system fallback, got %s", python)
	}
}

func TestPythonPathWithRelativeWorkingDirReturnsAbsolute(t *testing.T) {
	tmp := t.TempDir()
	pythonBin := filepath.Join(tmp, "wd", ".venv", "bin", "python")
	if err := os.MkdirAll(filepath.Dir(pythonBin), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(pythonBin, []byte("dummy"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(tmp)

	python := PythonPath("wd")
	if !filepath.IsAbs(python) {
		t.Fatalf("expected absolute path, got %s", python)
	}
	resolved, err := filepath.EvalSymlinks(python)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := filepath.EvalSymlinks(pythonBin)
	if err != nil {
		t.Fatal(err)
	}
	if resolved != expected {
		t.Fatalf("expected %s, got %s", expected, resolved)
	}
}
