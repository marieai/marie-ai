package local_runtime

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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

func TestEnsureEnvironmentSyncsUvProject(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	dir := t.TempDir()
	writeUvProject(t, dir)

	python, err := EnsureEnvironment(context.Background(), dir, "3.12", io.Discard)
	if err != nil {
		t.Fatalf("environment setup failed: %v", err)
	}
	if _, err := os.Stat(python); err != nil {
		t.Fatalf("venv python missing: %v", err)
	}
}

func TestEnsureEnvironmentDoesNotInheritParentVirtualEnv(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	dir := t.TempDir()
	writeUvProject(t, dir)
	parentVenv := filepath.Join(t.TempDir(), "parent-venv")
	t.Setenv("VIRTUAL_ENV", parentVenv)
	var logs bytes.Buffer

	python, err := EnsureEnvironment(context.Background(), dir, "3.12", &logs)
	if err != nil {
		t.Fatalf("environment setup failed: %v", err)
	}
	if strings.Contains(logs.String(), "VIRTUAL_ENV") {
		t.Fatalf("uv inherited parent virtual environment:\n%s", logs.String())
	}
	if filepath.Dir(filepath.Dir(python)) != filepath.Join(dir, ".venv") {
		t.Fatalf("expected plugin-local virtual environment, got %s", python)
	}
	if os.Getenv("VIRTUAL_ENV") != parentVenv {
		t.Fatal("parent virtual environment was modified")
	}
}

func TestEnsureEnvironmentRequiresLockForUvProject(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte(`[project]
name = "test-plugin"
version = "0.0.0"
requires-python = ">=3.12,<3.13"
`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "requirements.txt"), nil, 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := EnsureEnvironment(context.Background(), dir, "3.12", io.Discard)
	if !errors.Is(err, ErrEnvironmentSetup) {
		t.Fatalf("expected ErrEnvironmentSetup, got %v", err)
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

func writeUvProject(t *testing.T, dir string) {
	t.Helper()
	project := `[project]
name = "test-plugin"
version = "0.0.0"
requires-python = ">=3.12,<3.13"

[tool.uv]
package = false
`
	if err := os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte(project), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("uv", "lock", "--project", dir)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("uv lock failed: %v\n%s", err, output)
	}
}
