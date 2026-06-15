package backwards_invocation

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

var (
	ErrStorageInvalidName = errors.New("storage_invalid_name")
	ErrStorageValueTooBig = errors.New("storage_value_too_large")
)

const maxStorageValueBytes = 1 << 20

// Storage is a per-tenant, per-package key-value store backed by files under
// <root>/kv/<tenant>/<packageRef>/<sha256hex(key)>.bin.
type Storage struct {
	root string
	mu   sync.Mutex
}

func NewStorage(root string) *Storage {
	return &Storage{root: root}
}

func (s *Storage) Set(tenant, packageRef, key string, value []byte) error {
	if len(value) > maxStorageValueBytes {
		return fmt.Errorf("%w: %d bytes exceeds %d", ErrStorageValueTooBig, len(value), maxStorageValueBytes)
	}
	path, err := s.path(tenant, packageRef, key)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	temp, err := os.CreateTemp(dir, "kv-*.tmp")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	if _, err := temp.Write(value); err != nil {
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

func (s *Storage) Get(tenant, packageRef, key string) ([]byte, bool, error) {
	path, err := s.path(tenant, packageRef, key)
	if err != nil {
		return nil, false, err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	value, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	return value, true, nil
}

func (s *Storage) Delete(tenant, packageRef, key string) error {
	path, err := s.path(tenant, packageRef, key)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func (s *Storage) path(tenant, packageRef, key string) (string, error) {
	if err := validateStorageName(tenant); err != nil {
		return "", err
	}
	if err := validateStorageName(packageRef); err != nil {
		return "", err
	}
	digest := sha256.Sum256([]byte(key))
	return filepath.Join(s.root, "kv", tenant, packageRef, hex.EncodeToString(digest[:])+".bin"), nil
}

// validateStorageName mirrors plugins.validateName: tenant and packageRef were
// validated at install time, but the store re-checks before touching disk.
func validateStorageName(value string) error {
	if value == "" || value == "." || strings.Contains(value, "..") {
		return fmt.Errorf("%w: %q", ErrStorageInvalidName, value)
	}
	for _, char := range value {
		switch {
		case char >= 'a' && char <= 'z', char >= 'A' && char <= 'Z', char >= '0' && char <= '9', char == '_', char == '.', char == '-':
		default:
			return fmt.Errorf("%w: %q", ErrStorageInvalidName, value)
		}
	}
	return nil
}
