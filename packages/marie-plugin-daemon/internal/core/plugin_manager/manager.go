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
	install, err := m.Prepare(tenant, archive)
	if err != nil {
		return Install{}, err
	}
	if err := m.Activate(install); err != nil {
		return Install{}, err
	}
	return install, nil
}

// Prepare extracts an immutable version without changing the active install.
func (m *Manager) Prepare(tenant string, archive []byte) (Install, error) {
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

	if current, ok := m.get(tenant, packageRef); ok && current.Digest == digest {
		return current, nil
	}

	digestDir := packageRef + "@" + strings.ReplaceAll(digest, ":", "-")
	workingDir := filepath.Join(m.tenantDir(tenant), digestDir)
	install := Install{
		Tenant:     tenant,
		PackageRef: packageRef,
		Digest:     digest,
		WorkingDir: workingDir,
		State:      StateInstalled,
	}
	if info, err := os.Stat(workingDir); err == nil && info.IsDir() {
		return install, nil
	} else if err != nil && !errors.Is(err, os.ErrNotExist) {
		return Install{}, err
	}
	if err := os.MkdirAll(m.tenantDir(tenant), 0o755); err != nil {
		return Install{}, err
	}
	stagingDir, err := os.MkdirTemp(m.tenantDir(tenant), ".stage-")
	if err != nil {
		return Install{}, err
	}
	defer os.RemoveAll(stagingDir)
	if err := extractZip(archive, stagingDir); err != nil {
		return Install{}, err
	}
	if err := os.Rename(stagingDir, workingDir); err != nil {
		return Install{}, err
	}
	return install, nil
}

// Activate atomically selects a prepared version.
func (m *Manager) Activate(install Install) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return writeInstall(m.installPath(install.Tenant, install.PackageRef), install)
}

// Discard removes a failed or retired version, never the active installation.
func (m *Manager) Discard(install Install) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if current, ok := m.get(install.Tenant, install.PackageRef); ok && current.WorkingDir == install.WorkingDir {
		return nil
	}
	return os.RemoveAll(install.WorkingDir)
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
	if err := os.Remove(m.installPath(tenant, packageRef)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
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
	return writeInstall(m.installPath(tenant, packageRef), install)
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

	// New installs use one atomic selection record per package. Version-local
	// records remain readable for storage created by older daemon versions.
	paths := []string{}
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".json") {
			paths = append(paths, filepath.Join(m.tenantDir(tenant), entry.Name()))
		}
	}
	for _, entry := range entries {
		if entry.IsDir() {
			paths = append(paths, filepath.Join(m.tenantDir(tenant), entry.Name(), "install.json"))
		}
	}
	selected := map[string]bool{}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		install := Install{}
		if err := json.Unmarshal(data, &install); err != nil || selected[install.PackageRef] {
			continue
		}
		selected[install.PackageRef] = true
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

func (m *Manager) installPath(tenant, packageRef string) string {
	return filepath.Join(m.tenantDir(tenant), packageRef+".json")
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

func writeInstall(path string, install Install) error {
	data, err := json.MarshalIndent(install, "", "  ")
	if err != nil {
		return err
	}

	temp, err := os.CreateTemp(filepath.Dir(path), ".install-*.tmp")
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
	if err := os.Rename(tempPath, path); err != nil {
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
