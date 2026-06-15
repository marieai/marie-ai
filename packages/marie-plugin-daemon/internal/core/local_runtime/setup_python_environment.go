package local_runtime

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
)

var (
	ErrEnvironmentSetup = errors.New("environment_setup_failed")
)

var pythonVersionPattern = regexp.MustCompile(`^\d+\.\d+(\.\d+)?$`)

func EnsureEnvironment(ctx context.Context, workingDir string, pythonVersion string, logs io.Writer) (string, error) {
	if pythonVersion == "" {
		pythonVersion = "3.12"
	}
	if !pythonVersionPattern.MatchString(pythonVersion) {
		return "", fmt.Errorf("%w: invalid python version: %s", ErrEnvironmentSetup, pythonVersion)
	}

	absDir, err := filepath.Abs(workingDir)
	if err != nil {
		return "", fmt.Errorf("%w: resolve working dir: %v", ErrEnvironmentSetup, err)
	}

	pythonBin := filepath.Join(absDir, ".venv", "bin", "python")

	cmd := exec.CommandContext(ctx, "uv", "venv", "--python", pythonVersion, ".venv")
	cmd.Dir = absDir
	cmd.Stdout = logs
	cmd.Stderr = logs

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("%w: uv venv: %v", ErrEnvironmentSetup, err)
	}

	requirementsPath := filepath.Join(absDir, "requirements.txt")
	info, err := os.Stat(requirementsPath)
	if err == nil && info.Size() > 0 {
		cmd := exec.CommandContext(ctx, "uv", "pip", "install", "-r", "requirements.txt", "--python", pythonBin)
		cmd.Dir = absDir
		cmd.Stdout = logs
		cmd.Stderr = logs

		if err := cmd.Run(); err != nil {
			return "", fmt.Errorf("%w: uv pip install: %v", ErrEnvironmentSetup, err)
		}
	}

	return pythonBin, nil
}

func PythonPath(workingDir string) string {
	absDir, err := filepath.Abs(workingDir)
	if err != nil {
		return "python3"
	}
	pythonBin := filepath.Join(absDir, ".venv", "bin", "python")
	if _, err := os.Stat(pythonBin); err == nil {
		return pythonBin
	}
	return "python3"
}
