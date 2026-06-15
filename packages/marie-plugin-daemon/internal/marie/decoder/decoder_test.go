package decoder

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

const manifest = `apiVersion: marie.ai/v1alpha1
kind: ExtensionPackage
metadata:
  id: ext.test.minimal-tool
  author: marie
  name: minimal-tool
  version: 0.1.0
providers:
  - ref: provider/minimal
    type: tool_provider
`

func TestDecodeDirectory(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "marie-extension.yaml"), manifest)
	writeFile(t, filepath.Join(root, "README.md"), "# Minimal")
	writeFile(t, filepath.Join(root, "assets", "matcher.js"), "export default {}")

	result, err := DecodePath(root)
	if err != nil {
		t.Fatal(err)
	}

	if result.Identity.PackageRef != "ext.test.minimal-tool" {
		t.Fatalf("unexpected package ref: %s", result.Identity.PackageRef)
	}
	if result.Identity.Author != "marie" {
		t.Fatalf("unexpected author: %s", result.Identity.Author)
	}
	if len(result.Providers) != 1 {
		t.Fatalf("expected one provider, got %d", len(result.Providers))
	}
	if len(result.Readmes) != 1 {
		t.Fatalf("expected one readme, got %d", len(result.Readmes))
	}
	if len(result.Assets) != 1 {
		t.Fatalf("expected one asset, got %d", len(result.Assets))
	}
	if result.Verification.State != "unsigned" || result.Verification.Trusted {
		t.Fatalf("unexpected verification state: %#v", result.Verification)
	}
}

func TestDecodeMarieExtensionFixture(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime caller unavailable")
	}
	fixturePath := filepath.Clean(filepath.Join(
		filepath.Dir(file),
		"..",
		"..",
		"..",
		"..",
		"marie-extension",
		"tests",
		"fixtures",
		"minimal-tool",
	))

	result, err := DecodePath(fixturePath)
	if err != nil {
		t.Fatal(err)
	}

	if result.Identity.PackageRef != "ext.test.minimal-tool" {
		t.Fatalf("unexpected fixture package ref: %s", result.Identity.PackageRef)
	}
	if len(result.Providers) != 1 {
		t.Fatalf("expected one fixture provider, got %d", len(result.Providers))
	}
}

func TestDecodeZip(t *testing.T) {
	archive, err := ZipFixture(map[string][]byte{
		"package/marie-extension.yaml": []byte(manifest),
		"package/README.md":            []byte("# Minimal"),
	})
	if err != nil {
		t.Fatal(err)
	}

	zipPath := filepath.Join(t.TempDir(), "minimal.zip")
	if err := os.WriteFile(zipPath, archive, 0o644); err != nil {
		t.Fatal(err)
	}

	result, err := DecodePath(zipPath)
	if err != nil {
		t.Fatal(err)
	}

	if result.Identity.Name != "minimal-tool" {
		t.Fatalf("unexpected package name: %s", result.Identity.Name)
	}
	if result.Checksum == "" {
		t.Fatal("checksum missing")
	}
}

func TestRejectOversizedZipEntry(t *testing.T) {
	previousEntry, previousArchive := maxEntryBytes, maxArchiveBytes
	maxEntryBytes, maxArchiveBytes = 1024, 4096
	defer func() { maxEntryBytes, maxArchiveBytes = previousEntry, previousArchive }()

	archive, err := ZipFixture(map[string][]byte{
		"marie-extension.yaml": []byte(manifest),
		"payload.bin":          bytes.Repeat([]byte{0}, 2048),
	})
	if err != nil {
		t.Fatal(err)
	}

	zipPath := filepath.Join(t.TempDir(), "oversized.zip")
	if err := os.WriteFile(zipPath, archive, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := DecodePath(zipPath); !errors.Is(err, ErrPackageTooLarge) {
		t.Fatalf("expected package too large error, got %v", err)
	}
}

func TestRejectOversizedArchiveTotal(t *testing.T) {
	previousEntry, previousArchive := maxEntryBytes, maxArchiveBytes
	maxEntryBytes, maxArchiveBytes = 4096, 4096
	defer func() { maxEntryBytes, maxArchiveBytes = previousEntry, previousArchive }()

	archive, err := ZipFixture(map[string][]byte{
		"marie-extension.yaml": []byte(manifest),
		"a.bin":                bytes.Repeat([]byte{0}, 3000),
		"b.bin":                bytes.Repeat([]byte{0}, 3000),
	})
	if err != nil {
		t.Fatal(err)
	}

	zipPath := filepath.Join(t.TempDir(), "oversized-total.zip")
	if err := os.WriteFile(zipPath, archive, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := DecodePath(zipPath); !errors.Is(err, ErrPackageTooLarge) {
		t.Fatalf("expected package too large error, got %v", err)
	}
}

func TestRejectDifyPackage(t *testing.T) {
	sourcePath := filepath.Join(t.TempDir(), "plugin.difypkg")
	if err := os.WriteFile(sourcePath, []byte("not a marie package"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := DecodePath(sourcePath)
	if !errors.Is(err, ErrUnsupportedPackageFormat) {
		t.Fatalf("expected unsupported package error, got %v", err)
	}
}

func TestRejectMultipleManifests(t *testing.T) {
	archive, err := ZipFixture(map[string][]byte{
		"a/marie-extension.yaml": []byte(manifest),
		"b/marie-extension.yaml": []byte(manifest),
	})
	if err != nil {
		t.Fatal(err)
	}

	zipPath := filepath.Join(t.TempDir(), "multiple.zip")
	if err := os.WriteFile(zipPath, archive, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := DecodePath(zipPath); err == nil {
		t.Fatal("expected multiple manifest error")
	}
}

func writeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
