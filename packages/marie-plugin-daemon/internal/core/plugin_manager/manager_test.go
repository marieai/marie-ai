package plugin_manager

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/decoder"
)

func TestInstallExtractsPackageAndPersistsState(t *testing.T) {
	root := t.TempDir()
	manager := NewManager(root)
	zipBytes := zipFixturePlugin(t)

	install, err := manager.Install("org1__ws1", zipBytes)
	if err != nil {
		t.Fatalf("install failed: %v", err)
	}
	if install.PackageRef == "" || install.Digest == "" {
		t.Fatalf("missing identity: %+v", install)
	}
	if install.State != "installed" {
		t.Fatalf("unexpected initial state: %q", install.State)
	}
	expectedDir := filepath.Join(root, pluginRuntimeDir, "org1__ws1")
	if filepath.Dir(install.WorkingDir) != expectedDir {
		t.Fatalf("working dir = %q, want child of %q", install.WorkingDir, expectedDir)
	}
	if _, err := os.Stat(filepath.Join(install.WorkingDir, "main.py")); err != nil {
		t.Fatalf("source not extracted into working dir: %v", err)
	}

	listed := NewManager(root).List("org1__ws1")
	if len(listed) != 1 || listed[0].Digest != install.Digest {
		t.Fatalf("install state not persisted: %+v", listed)
	}
}

func TestInstallRejectsPathTraversal(t *testing.T) {
	root := t.TempDir()
	if _, err := NewManager(root).Install("org1__ws1", zipWithEntry(t, "../escape.py")); err == nil {
		t.Fatal("expected traversal rejection")
	}
}

func TestReinstallReplacesPreviousDigest(t *testing.T) {
	root := t.TempDir()
	manager := NewManager(root)

	first, err := manager.Install("org1__ws1", zipFixturePlugin(t))
	if err != nil {
		t.Fatalf("first install failed: %v", err)
	}

	files := fixtureFiles(t)
	files["extra.py"] = []byte("print('extra')\n")
	second, err := manager.Install("org1__ws1", zipFromFiles(t, files))
	if err != nil {
		t.Fatalf("second install failed: %v", err)
	}
	if second.Digest == first.Digest {
		t.Fatalf("expected new digest, got %q twice", second.Digest)
	}
	if _, err := os.Stat(first.WorkingDir); !os.IsNotExist(err) {
		t.Fatalf("previous working dir not removed: %v", err)
	}

	listed := manager.List("org1__ws1")
	if len(listed) != 1 || listed[0].Digest != second.Digest {
		t.Fatalf("expected single install with new digest: %+v", listed)
	}
}

func TestReinstallSameArchiveIsIdempotent(t *testing.T) {
	manager := NewManager(t.TempDir())
	zipBytes := zipFixturePlugin(t)

	first, err := manager.Install("org1__ws1", zipBytes)
	if err != nil {
		t.Fatalf("first install failed: %v", err)
	}
	second, err := manager.Install("org1__ws1", zipBytes)
	if err != nil {
		t.Fatalf("second install failed: %v", err)
	}
	if second.Digest != first.Digest || second.WorkingDir != first.WorkingDir {
		t.Fatalf("expected identical install, got %+v vs %+v", first, second)
	}
	if _, err := os.Stat(filepath.Join(second.WorkingDir, "main.py")); err != nil {
		t.Fatalf("source missing after reinstall: %v", err)
	}
	if listed := manager.List("org1__ws1"); len(listed) != 1 {
		t.Fatalf("expected single install: %+v", listed)
	}
}

func TestInstallRejectsMissingPackageRef(t *testing.T) {
	manifest := []byte("apiVersion: marie.ai/v1alpha1\nkind: ExtensionPackage\nmetadata:\n  name: no-id\n  author: marie\n  version: 0.0.1\n")
	archive := zipFromFiles(t, map[string][]byte{
		"marie-extension.yaml": manifest,
		"main.py":              []byte("print('x')\n"),
	})

	_, err := NewManager(t.TempDir()).Install("org1__ws1", archive)
	if !errors.Is(err, ErrInvalidPackage) {
		t.Fatalf("expected invalid_package, got %v", err)
	}
}

func TestInstallRejectsRootInstallJSONEntry(t *testing.T) {
	files := fixtureFiles(t)
	files["install.json"] = []byte("{\"state\":\"forged\"}")

	_, err := NewManager(t.TempDir()).Install("org1__ws1", zipFromFiles(t, files))
	if !errors.Is(err, ErrUnsafeArchivePath) {
		t.Fatalf("expected reserved install.json rejection, got %v", err)
	}
}

func TestRejectsHostileNames(t *testing.T) {
	manager := NewManager(t.TempDir())
	zipBytes := zipFixturePlugin(t)

	for _, tenant := range []string{"", "../x", "a/b", "a\\b", "org1 ws1"} {
		if _, err := manager.Install(tenant, zipBytes); !errors.Is(err, ErrInvalidName) {
			t.Fatalf("expected invalid_name for tenant %q, got %v", tenant, err)
		}
	}
	if err := manager.Remove("org1__ws1", "ext/echo"); !errors.Is(err, ErrInvalidName) {
		t.Fatalf("expected invalid_name for ref with slash, got %v", err)
	}
	if err := manager.SetState("org1__ws1", "ext..echo", "ready"); !errors.Is(err, ErrInvalidName) {
		t.Fatalf("expected invalid_name for ref with dotdot, got %v", err)
	}
}

func TestConcurrentInstallSetStateList(t *testing.T) {
	manager := NewManager(t.TempDir())
	zipBytes := zipFixturePlugin(t)

	install, err := manager.Install("org1__ws1", zipBytes)
	if err != nil {
		t.Fatalf("seed install failed: %v", err)
	}

	wg := sync.WaitGroup{}
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for round := 0; round < 5; round++ {
				switch worker % 3 {
				case 0:
					if _, err := manager.Install("org1__ws1", zipBytes); err != nil {
						t.Errorf("concurrent install failed: %v", err)
					}
				case 1:
					if err := manager.SetState("org1__ws1", install.PackageRef, "ready"); err != nil {
						t.Errorf("concurrent set state failed: %v", err)
					}
				default:
					manager.List("org1__ws1")
				}
			}
		}(worker)
	}
	wg.Wait()

	listed := manager.List("org1__ws1")
	if len(listed) != 1 || listed[0].PackageRef != install.PackageRef || listed[0].Digest != install.Digest {
		t.Fatalf("inconsistent final state: %+v", listed)
	}
	if listed[0].State != "installed" && listed[0].State != "ready" {
		t.Fatalf("unexpected final state: %q", listed[0].State)
	}
}

func TestRemoveDeletesInstall(t *testing.T) {
	root := t.TempDir()
	manager := NewManager(root)

	install, err := manager.Install("org1__ws1", zipFixturePlugin(t))
	if err != nil {
		t.Fatalf("install failed: %v", err)
	}
	if err := manager.Remove("org1__ws1", install.PackageRef); err != nil {
		t.Fatalf("remove failed: %v", err)
	}
	if _, ok := manager.Get("org1__ws1", install.PackageRef); ok {
		t.Fatal("expected install gone after remove")
	}
	if err := manager.Remove("org1__ws1", install.PackageRef); err == nil {
		t.Fatal("expected error removing missing install")
	}
}

func TestSetStateRoundTrip(t *testing.T) {
	root := t.TempDir()
	manager := NewManager(root)

	install, err := manager.Install("org1__ws1", zipFixturePlugin(t))
	if err != nil {
		t.Fatalf("install failed: %v", err)
	}
	if err := manager.SetState("org1__ws1", install.PackageRef, "ready"); err != nil {
		t.Fatalf("set state failed: %v", err)
	}

	updated, ok := NewManager(root).Get("org1__ws1", install.PackageRef)
	if !ok || updated.State != "ready" {
		t.Fatalf("state not persisted: %+v ok=%v", updated, ok)
	}
	if err := manager.SetState("org1__ws1", "ext.test.missing", "ready"); err == nil {
		t.Fatal("expected error for unknown package")
	}
}

func TestSetStateRejectsUnknownState(t *testing.T) {
	manager := NewManager(t.TempDir())
	install, err := manager.Install("org1__ws1", zipFixturePlugin(t))
	if err != nil {
		t.Fatalf("install failed: %v", err)
	}
	if err := manager.SetState("org1__ws1", install.PackageRef, "forged"); !errors.Is(err, ErrInvalidState) {
		t.Fatalf("expected invalid_state, got %v", err)
	}
	for _, state := range []string{StateInstalled, StateStarting, StateReady, StateUnresponsive, StateStopped, StateFailed} {
		if err := manager.SetState("org1__ws1", install.PackageRef, state); err != nil {
			t.Fatalf("expected %q accepted, got %v", state, err)
		}
	}
}

func TestExtractZipRejectsOversizedContent(t *testing.T) {
	previous := maxExtractedBytes
	maxExtractedBytes = 10
	defer func() { maxExtractedBytes = previous }()

	err := extractZip(zipFromFiles(t, map[string][]byte{"big.bin": make([]byte, 32)}), t.TempDir())
	if !errors.Is(err, ErrArchiveTooLarge) {
		t.Fatalf("expected archive_too_large, got %v", err)
	}
}

func TestCountSumsAcrossTenants(t *testing.T) {
	manager := NewManager(t.TempDir())
	if manager.Count() != 0 {
		t.Fatalf("expected zero installs, got %d", manager.Count())
	}
	if _, err := manager.Install("org1__ws1", zipFixturePlugin(t)); err != nil {
		t.Fatalf("install failed: %v", err)
	}
	if _, err := manager.Install("org2__ws2", zipFixturePlugin(t)); err != nil {
		t.Fatalf("install failed: %v", err)
	}
	if manager.Count() != 2 {
		t.Fatalf("expected 2 installs, got %d", manager.Count())
	}
}

func TestExtractZipRejectsTraversalEntries(t *testing.T) {
	for _, name := range []string{"../escape.py", "nested/../../escape.py", "/etc/escape.py"} {
		err := extractZip(zipFromFiles(t, map[string][]byte{name: []byte("x")}), t.TempDir())
		if !errors.Is(err, ErrUnsafeArchivePath) {
			t.Fatalf("expected unsafe path rejection for %q, got %v", name, err)
		}
	}
}

func TestGetUnknownPackageReturnsFalse(t *testing.T) {
	manager := NewManager(t.TempDir())
	if _, ok := manager.Get("org1__ws1", "ext.test.missing"); ok {
		t.Fatal("expected miss for unknown package")
	}
}

func fixtureFiles(t *testing.T) map[string][]byte {
	t.Helper()
	root := filepath.Join("..", "..", "..", "testdata", "fixture-plugin")
	files := map[string][]byte{}
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
		files[filepath.ToSlash(rel)] = data
		return nil
	})
	if err != nil {
		t.Fatalf("read fixture plugin: %v", err)
	}
	return files
}

func zipFromFiles(t *testing.T, files map[string][]byte) []byte {
	t.Helper()
	zipBytes, err := decoder.ZipFixture(files)
	if err != nil {
		t.Fatalf("zip fixture: %v", err)
	}
	return zipBytes
}

func zipFixturePlugin(t *testing.T) []byte {
	t.Helper()
	return zipFromFiles(t, fixtureFiles(t))
}

func zipWithEntry(t *testing.T, name string) []byte {
	t.Helper()
	manifest, err := os.ReadFile(filepath.Join("..", "..", "..", "testdata", "fixture-plugin", "marie-extension.yaml"))
	if err != nil {
		t.Fatalf("read fixture manifest: %v", err)
	}
	return zipFromFiles(t, map[string][]byte{
		"marie-extension.yaml": manifest,
		name:                   []byte("print('escape')\n"),
	})
}
