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
	"strings"
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
	pyprojectPath := filepath.Join(absDir, "pyproject.toml")
	lockPath := filepath.Join(absDir, "uv.lock")

	_, pyprojectErr := os.Stat(pyprojectPath)
	_, lockErr := os.Stat(lockPath)
	if pyprojectErr == nil {
		if errors.Is(lockErr, os.ErrNotExist) {
			return "", fmt.Errorf("%w: uv.lock is required with pyproject.toml", ErrEnvironmentSetup)
		}
		if lockErr != nil {
			return "", fmt.Errorf("%w: inspect uv.lock: %v", ErrEnvironmentSetup, lockErr)
		}

		cmd := uvCommand(ctx, "sync", "--locked", "--no-dev", "--python", pythonVersion)
		cmd.Dir = absDir
		cmd.Stdout = logs
		cmd.Stderr = logs
		if err := cmd.Run(); err != nil {
			return "", fmt.Errorf("%w: uv sync: %v", ErrEnvironmentSetup, err)
		}
	} else {
		if !errors.Is(pyprojectErr, os.ErrNotExist) {
			return "", fmt.Errorf("%w: inspect pyproject.toml: %v", ErrEnvironmentSetup, pyprojectErr)
		}
		if lockErr == nil {
			return "", fmt.Errorf("%w: pyproject.toml is required with uv.lock", ErrEnvironmentSetup)
		}
		if !errors.Is(lockErr, os.ErrNotExist) {
			return "", fmt.Errorf("%w: inspect uv.lock: %v", ErrEnvironmentSetup, lockErr)
		}

		if _, err := os.Stat(pythonBin); errors.Is(err, os.ErrNotExist) {
			cmd := uvCommand(ctx, "venv", "--python", pythonVersion, ".venv")
			cmd.Dir = absDir
			cmd.Stdout = logs
			cmd.Stderr = logs

			if err := cmd.Run(); err != nil {
				return "", fmt.Errorf("%w: uv venv: %v", ErrEnvironmentSetup, err)
			}
		} else if err != nil {
			return "", fmt.Errorf("%w: inspect venv: %v", ErrEnvironmentSetup, err)
		}

		requirementsPath := filepath.Join(absDir, "requirements.txt")
		info, err := os.Stat(requirementsPath)
		if err == nil && info.Size() > 0 {
			cmd := uvCommand(ctx, "pip", "install", "-r", "requirements.txt", "--python", pythonBin)
			cmd.Dir = absDir
			cmd.Stdout = logs
			cmd.Stderr = logs

			if err := cmd.Run(); err != nil {
				return "", fmt.Errorf("%w: uv pip install: %v", ErrEnvironmentSetup, err)
			}
		}
	}

	cmd := uvCommand(ctx, "pip", "check", "--python", pythonBin)
	cmd.Dir = absDir
	cmd.Stdout = logs
	cmd.Stderr = logs
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("%w: uv pip check: %v", ErrEnvironmentSetup, err)
	}

	return pythonBin, nil
}

func uvCommand(ctx context.Context, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, "uv", args...)
	for _, value := range os.Environ() {
		if !strings.HasPrefix(value, "VIRTUAL_ENV=") {
			cmd.Env = append(cmd.Env, value)
		}
	}
	return cmd
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
