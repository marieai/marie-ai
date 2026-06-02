package decoder

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

var ErrUnsupportedPackageFormat = errors.New("unsupported package format")

type PackageIdentity struct {
	PackageRef string `json:"packageRef"`
	Author     string `json:"author"`
	Name       string `json:"name"`
	Version    string `json:"version"`
}

type VerificationState struct {
	State     string `json:"state"`
	Trusted   bool   `json:"trusted"`
	Signature string `json:"signatureDigest,omitempty"`
}

type InventoryEntry struct {
	Path        string `json:"path"`
	Size        int64  `json:"size"`
	Sha256      string `json:"sha256"`
	ContentType string `json:"contentType"`
}

type DecodeResponse struct {
	Identity     PackageIdentity   `json:"identity"`
	Checksum     string            `json:"checksum"`
	Verification VerificationState `json:"verification"`
	Manifest     map[string]any    `json:"manifest"`
	Providers    []any             `json:"providers"`
	Assets       []InventoryEntry  `json:"assets"`
	Readmes      []InventoryEntry  `json:"readmes"`
	Warnings     []string          `json:"warnings"`
	Errors       []string          `json:"errors"`
}

type packageFile struct {
	path string
	data []byte
}

func DecodePath(sourcePath string) (*DecodeResponse, error) {
	if strings.EqualFold(filepath.Ext(sourcePath), ".difypkg") {
		return nil, fmt.Errorf("%w: .difypkg is converter input, not a Marie runtime package", ErrUnsupportedPackageFormat)
	}

	stat, err := os.Stat(sourcePath)
	if err != nil {
		return nil, err
	}

	if stat.IsDir() {
		return decodeFiles(readDirectory(sourcePath))
	}

	return decodeFiles(readZip(sourcePath))
}

func decodeFiles(files []packageFile, readErr error) (*DecodeResponse, error) {
	if readErr != nil {
		return nil, readErr
	}

	sort.Slice(files, func(left, right int) bool {
		return files[left].path < files[right].path
	})

	manifestFiles := make([]packageFile, 0, 1)
	for _, file := range files {
		if filepath.Base(file.path) == "marie-extension.yaml" {
			manifestFiles = append(manifestFiles, file)
		}
	}

	if len(manifestFiles) != 1 {
		return nil, fmt.Errorf("expected exactly one marie-extension.yaml, found %d", len(manifestFiles))
	}

	manifest := map[string]any{}
	if err := yaml.Unmarshal(manifestFiles[0].data, &manifest); err != nil {
		return nil, err
	}
	manifest = normalizeMap(manifest)

	response := &DecodeResponse{
		Identity:     identityFromManifest(manifest),
		Checksum:     checksum(files),
		Verification: VerificationState{State: "unsigned", Trusted: false},
		Manifest:     manifest,
		Providers:    listField(manifest["providers"]),
		Warnings:     identityWarnings(manifest),
		Errors:       []string{},
	}

	for _, file := range files {
		if filepath.Base(file.path) == "marie-extension.yaml" {
			continue
		}
		entry := inventoryEntry(file)
		if isReadme(file.path) {
			response.Readmes = append(response.Readmes, entry)
		} else {
			response.Assets = append(response.Assets, entry)
		}
	}

	return response, nil
}

func readDirectory(root string) ([]packageFile, error) {
	files := []packageFile{}
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}

		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		files = append(files, packageFile{path: filepath.ToSlash(rel), data: data})
		return nil
	})
	return files, err
}

func readZip(sourcePath string) ([]packageFile, error) {
	reader, err := zip.OpenReader(sourcePath)
	if err != nil {
		return nil, err
	}
	defer reader.Close()

	files := []packageFile{}
	for _, file := range reader.File {
		if file.FileInfo().IsDir() {
			continue
		}

		name := filepath.ToSlash(filepath.Clean(file.Name))
		if strings.HasPrefix(name, "../") || strings.HasPrefix(name, "/") {
			return nil, fmt.Errorf("unsafe package path: %s", file.Name)
		}

		opened, err := file.Open()
		if err != nil {
			return nil, err
		}
		data, readErr := io.ReadAll(opened)
		closeErr := opened.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		files = append(files, packageFile{path: name, data: data})
	}
	return files, nil
}

func identityFromManifest(manifest map[string]any) PackageIdentity {
	metadata := mapField(manifest["metadata"])
	name := stringField(metadata["name"])
	author := stringField(metadata["author"])
	return PackageIdentity{
		PackageRef: stringField(metadata["id"]),
		Author:     author,
		Name:       name,
		Version:    stringField(metadata["version"]),
	}
}

func identityWarnings(manifest map[string]any) []string {
	identity := identityFromManifest(manifest)
	warnings := []string{}
	if identity.PackageRef == "" {
		warnings = append(warnings, "metadata.id is missing")
	}
	if identity.Name == "" {
		warnings = append(warnings, "metadata.name is missing")
	}
	if identity.Version == "" {
		warnings = append(warnings, "metadata.version is missing")
	}
	return warnings
}

func checksum(files []packageFile) string {
	hash := sha256.New()
	for _, file := range files {
		hash.Write([]byte(file.path))
		hash.Write([]byte{0})
		hash.Write(file.data)
		hash.Write([]byte{0})
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil))
}

func inventoryEntry(file packageFile) InventoryEntry {
	hash := sha256.Sum256(file.data)
	return InventoryEntry{
		Path:        file.path,
		Size:        int64(len(file.data)),
		Sha256:      "sha256:" + hex.EncodeToString(hash[:]),
		ContentType: contentType(file.path),
	}
}

func normalizeMap(value map[string]any) map[string]any {
	normalized := map[string]any{}
	for key, item := range value {
		normalized[key] = normalizeValue(item)
	}
	return normalized
}

func normalizeValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return normalizeMap(typed)
	case []any:
		items := make([]any, 0, len(typed))
		for _, item := range typed {
			items = append(items, normalizeValue(item))
		}
		return items
	default:
		return typed
	}
}

func mapField(value any) map[string]any {
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	return map[string]any{}
}

func listField(value any) []any {
	if typed, ok := value.([]any); ok {
		return typed
	}
	return []any{}
}

func stringField(value any) string {
	if typed, ok := value.(string); ok {
		return typed
	}
	return ""
}

func isReadme(path string) bool {
	return strings.HasPrefix(strings.ToLower(filepath.Base(path)), "readme")
}

func contentType(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".md":
		return "text/markdown"
	case ".yaml", ".yml":
		return "application/yaml"
	case ".json":
		return "application/json"
	case ".js":
		return "text/javascript"
	case ".py":
		return "text/x-python"
	default:
		return "application/octet-stream"
	}
}

func ZipFixture(files map[string][]byte) ([]byte, error) {
	buffer := bytes.NewBuffer(nil)
	writer := zip.NewWriter(buffer)
	paths := make([]string, 0, len(files))
	for path := range files {
		paths = append(paths, path)
	}
	sort.Strings(paths)

	for _, path := range paths {
		file, err := writer.Create(path)
		if err != nil {
			return nil, err
		}
		if _, err := file.Write(files[path]); err != nil {
			return nil, err
		}
	}

	if err := writer.Close(); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}
