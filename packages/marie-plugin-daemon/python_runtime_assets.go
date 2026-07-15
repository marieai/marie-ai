// Package plugindaemon exposes assets shared by the daemon's internal packages.
package plugindaemon

import (
	"embed"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
)

// ErrPythonRuntimeSetup classifies failures while materializing the shared
// Python runtime.
var ErrPythonRuntimeSetup = errors.New("python_runtime_setup_failed")

//go:embed python_runtime/marie_plugins/runtime/*.py
var pythonRuntime embed.FS

// PreparePythonRuntime materializes the embedded Python runtime for one plugin
// environment.
func PreparePythonRuntime(workingDir string) (string, error) {
	absDir, err := filepath.Abs(workingDir)
	if err != nil {
		return "", fmt.Errorf("%w: resolve working dir: %v", ErrPythonRuntimeSetup, err)
	}

	runtimeRoot := filepath.Join(absDir, ".venv", "marie-plugin-runtime")
	packageRoot := filepath.Join(runtimeRoot, "marie_plugins", "runtime")
	if err := os.MkdirAll(packageRoot, 0o755); err != nil {
		return "", fmt.Errorf("%w: create runtime directory: %v", ErrPythonRuntimeSetup, err)
	}

	const embeddedRoot = "python_runtime/marie_plugins/runtime"
	entries, err := pythonRuntime.ReadDir(embeddedRoot)
	if err != nil {
		return "", fmt.Errorf("%w: read embedded runtime: %v", ErrPythonRuntimeSetup, err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		data, err := pythonRuntime.ReadFile(path.Join(embeddedRoot, entry.Name()))
		if err != nil {
			return "", fmt.Errorf("%w: read %s: %v", ErrPythonRuntimeSetup, entry.Name(), err)
		}
		if err := os.WriteFile(filepath.Join(packageRoot, entry.Name()), data, 0o644); err != nil {
			return "", fmt.Errorf("%w: write %s: %v", ErrPythonRuntimeSetup, entry.Name(), err)
		}
	}

	return runtimeRoot, nil
}
