package plugin_manager

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/decoder"
)

var (
	ErrInstallNotFound   = errors.New("install_not_found")
	ErrUnsafeArchivePath = errors.New("unsafe_archive_path")
	ErrInvalidPackage    = errors.New("invalid_package")
	ErrInvalidName       = errors.New("invalid_name")
	ErrInvalidState      = errors.New("invalid_state")
	ErrArchiveTooLarge   = errors.New("archive_too_large")
)

const (
	StateInstalled    = "installed"
	StateStarting     = "starting"
	StateReady        = "ready"
	StateUnresponsive = "unresponsive"
	StateStopped      = "stopped"
	StateFailed       = "failed"
	pluginRuntimeDir  = "plugin-runtime"
)

var maxExtractedBytes int64 = 1 << 30

type Install struct {
	Tenant     string `json:"tenant"`
	PackageRef string `json:"packageRef"`
	Digest     string `json:"digest"`
	WorkingDir string `json:"workingDir"`
	State      string `json:"state"` // installed|starting|ready|unresponsive|stopped|failed
}

type Manager struct {
	root string
	mu   sync.Mutex
}

func NewManager(root string) *Manager {
	return &Manager{root: root}
}

func (m *Manager) Install(tenant string, archive []byte) (Install, error) {
	if err := validateName(tenant); err != nil {
		return Install{}, err
	}

	packageRef, digest, err := Inspect(archive)
	if err != nil {
		return Install{}, err
	}
	if err := validateName(packageRef); err != nil {
		return Install{}, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, previous := range m.installDirs(tenant, packageRef) {
		if err := os.RemoveAll(previous); err != nil {
			return Install{}, err
		}
	}

	digestDir := packageRef + "@" + strings.ReplaceAll(digest, ":", "-")
	workingDir := filepath.Join(m.tenantDir(tenant), digestDir)
	if err := extractZip(archive, workingDir); err != nil {
		os.RemoveAll(workingDir)
		return Install{}, err
	}

	install := Install{
		Tenant:     tenant,
		PackageRef: packageRef,
		Digest:     digest,
		WorkingDir: workingDir,
		State:      StateInstalled,
	}
	if err := writeInstall(install); err != nil {
		os.RemoveAll(workingDir)
		return Install{}, err
	}
	return install, nil
}

// Inspect decodes an archive's identity without installing it.
func Inspect(archive []byte) (packageRef, digest string, err error) {
	temp, err := os.CreateTemp("", "marie-plugin-*.zip")
	if err != nil {
		return "", "", err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)

	if _, err := temp.Write(archive); err != nil {
		temp.Close()
		return "", "", err
	}
	if err := temp.Close(); err != nil {
		return "", "", err
	}

	decoded, err := decoder.DecodePath(tempPath)
	if err != nil {
		return "", "", err
	}
	if decoded.Identity.PackageRef == "" {
		return "", "", fmt.Errorf("%w: missing metadata.id", ErrInvalidPackage)
	}
	return decoded.Identity.PackageRef, decoded.Checksum, nil
}

func (m *Manager) List(tenant string) []Install {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.list(tenant)
}

func (m *Manager) Get(tenant, packageRef string) (Install, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.get(tenant, packageRef)
}

func (m *Manager) Remove(tenant, packageRef string) error {
	if err := validateName(tenant); err != nil {
		return err
	}
	if err := validateName(packageRef); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	dirs := m.installDirs(tenant, packageRef)
	if len(dirs) == 0 {
		return fmt.Errorf("%w: %s", ErrInstallNotFound, packageRef)
	}
	for _, dir := range dirs {
		if err := os.RemoveAll(dir); err != nil {
			return err
		}
	}
	return nil
}

func (m *Manager) SetState(tenant, packageRef, state string) error {
	if err := validateName(tenant); err != nil {
		return err
	}
	if err := validateName(packageRef); err != nil {
		return err
	}
	switch state {
	case StateInstalled, StateStarting, StateReady, StateUnresponsive, StateStopped, StateFailed:
	default:
		return fmt.Errorf("%w: %q", ErrInvalidState, state)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	install, ok := m.get(tenant, packageRef)
	if !ok {
		return fmt.Errorf("%w: %s", ErrInstallNotFound, packageRef)
	}
	install.State = state
	return writeInstall(install)
}

func (m *Manager) Count() int {
	m.mu.Lock()
	defer m.mu.Unlock()

	entries, err := os.ReadDir(filepath.Join(m.root, pluginRuntimeDir))
	if err != nil {
		return 0
	}
	total := 0
	for _, entry := range entries {
		if entry.IsDir() {
			total += len(m.list(entry.Name()))
		}
	}
	return total
}

func (m *Manager) list(tenant string) []Install {
	installs := []Install{}
	if validateName(tenant) != nil {
		return installs
	}
	entries, err := os.ReadDir(m.tenantDir(tenant))
	if err != nil {
		return installs
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(m.tenantDir(tenant), entry.Name(), "install.json"))
		if err != nil {
			continue
		}
		install := Install{}
		if err := json.Unmarshal(data, &install); err != nil {
			continue
		}
		installs = append(installs, install)
	}

	sort.Slice(installs, func(left, right int) bool {
		return installs[left].PackageRef < installs[right].PackageRef
	})
	return installs
}

func (m *Manager) get(tenant, packageRef string) (Install, bool) {
	for _, install := range m.list(tenant) {
		if install.PackageRef == packageRef {
			return install, true
		}
	}
	return Install{}, false
}

func (m *Manager) tenantDir(tenant string) string {
	return filepath.Join(m.root, pluginRuntimeDir, tenant)
}

func (m *Manager) installDirs(tenant, packageRef string) []string {
	entries, err := os.ReadDir(m.tenantDir(tenant))
	if err != nil {
		return nil
	}

	prefix := packageRef + "@"
	dirs := []string{}
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), prefix) {
			dirs = append(dirs, filepath.Join(m.tenantDir(tenant), entry.Name()))
		}
	}
	return dirs
}

func extractZip(archive []byte, targetDir string) error {
	reader, err := zip.NewReader(bytes.NewReader(archive), int64(len(archive)))
	if err != nil {
		return err
	}
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return err
	}

	remaining := maxExtractedBytes
	for _, file := range reader.File {
		if file.FileInfo().IsDir() {
			continue
		}

		name := filepath.Clean(filepath.FromSlash(file.Name))
		destination := filepath.Join(targetDir, name)
		rel, err := filepath.Rel(targetDir, destination)
		if err != nil || filepath.IsAbs(name) || hasDotDotSegment(name) || hasDotDotSegment(rel) {
			return fmt.Errorf("%w: %s", ErrUnsafeArchivePath, file.Name)
		}
		if name == "install.json" {
			return fmt.Errorf("%w: install.json is reserved", ErrUnsafeArchivePath)
		}

		if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
			return err
		}
		if err := extractFile(file, destination, &remaining); err != nil {
			return err
		}
	}
	return nil
}

func extractFile(file *zip.File, destination string, remaining *int64) error {
	opened, err := file.Open()
	if err != nil {
		return err
	}
	defer opened.Close()

	target, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	written, err := io.Copy(target, io.LimitReader(opened, *remaining+1))
	*remaining -= written
	if err != nil {
		target.Close()
		return err
	}
	if *remaining < 0 {
		target.Close()
		return fmt.Errorf("%w: uncompressed content exceeds %d bytes", ErrArchiveTooLarge, maxExtractedBytes)
	}
	return target.Close()
}

func writeInstall(install Install) error {
	data, err := json.MarshalIndent(install, "", "  ")
	if err != nil {
		return err
	}

	temp, err := os.CreateTemp(install.WorkingDir, "install-*.json.tmp")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	if _, err := temp.Write(data); err != nil {
		temp.Close()
		os.Remove(tempPath)
		return err
	}
	if err := temp.Close(); err != nil {
		os.Remove(tempPath)
		return err
	}
	if err := os.Chmod(tempPath, 0o644); err != nil {
		os.Remove(tempPath)
		return err
	}
	if err := os.Rename(tempPath, filepath.Join(install.WorkingDir, "install.json")); err != nil {
		os.Remove(tempPath)
		return err
	}
	return nil
}

func hasDotDotSegment(path string) bool {
	for _, segment := range strings.Split(path, string(filepath.Separator)) {
		if segment == ".." {
			return true
		}
	}
	return false
}

func validateName(value string) error {
	if value == "" || value == "." || strings.Contains(value, "..") {
		return fmt.Errorf("%w: %q", ErrInvalidName, value)
	}
	for _, char := range value {
		switch {
		case char >= 'a' && char <= 'z', char >= 'A' && char <= 'Z', char >= '0' && char <= '9', char == '_', char == '.', char == '-':
		default:
			return fmt.Errorf("%w: %q", ErrInvalidName, value)
		}
	}
	return nil
}
