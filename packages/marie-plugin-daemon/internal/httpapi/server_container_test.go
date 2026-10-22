package httpapi

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestContainerBuildContext(t *testing.T) {
	if os.Getenv("MARIE_PLUGIN_DAEMON_TEST_IMAGE") == "" {
		t.Skip("MARIE_PLUGIN_DAEMON_TEST_IMAGE is not set")
	}
	ignore, err := os.ReadFile("../../.dockerignore")
	if err != nil {
		t.Fatal(err)
	}
	buildContext, exported := t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(buildContext, ".dockerignore"), ignore, 0600); err != nil {
		t.Fatal(err)
	}
	files := map[string]bool{
		"go.mod":                   true,
		"go.sum":                   true,
		"python_runtime_assets.go": true,
		"cmd/server/main.go":       true,
		"internal/core/local_runtime/instance.go":               true,
		"pkg/client/client.go":                                  true,
		"python_runtime/marie_plugins/runtime/__init__.py":      true,
		"credentials.json":                                      false,
		"internal/local/credentials.json":                       false,
		"cmd/server/venv/bin/python":                            false,
		"pkg/client/.netrc":                                     false,
		"internal/local/cache/entry":                            false,
		"internal/core/local_runtime/instance_test.go":          false,
		"internal/testdata/fixture.go":                          false,
		"python_runtime/marie_plugins/runtime/credentials.json": false,
		"python_runtime/.venv/lib/module.py":                    false,
		"docker/serverless.dockerfile":                          false,
	}
	for name := range files {
		path := filepath.Join(buildContext, name)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("harmless build-context sentinel\n"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "docker", "build", "--network=none", "--output", "type=local,dest="+exported, "--file", "-", buildContext)
	command.Stdin = strings.NewReader("FROM scratch\nCOPY . /\n")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("export synthetic build context: %v: %s", err, output)
	}
	for name, included := range files {
		_, err := os.Stat(filepath.Join(exported, name))
		if err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
		if got := err == nil; got != included {
			t.Errorf("build context includes %s: got %t, want %t", name, got, included)
		}
	}
}

// Opt-in: the image must be built locally; this test never pulls or builds it.
func TestContainerPluginLifecycle(t *testing.T) {
	image := os.Getenv("MARIE_PLUGIN_DAEMON_TEST_IMAGE")
	if image == "" {
		t.Skip("MARIE_PLUGIN_DAEMON_TEST_IMAGE is not set")
	}
	t.Run("non_root_image_default", func(t *testing.T) {
		user := containerCommand(t, nil, "image", "inspect", "--format", "{{.Config.User}}", image)
		if user != "10001:10001" {
			t.Fatalf("expected unprivileged image user 10001:10001, got %q", user)
		}
	})
	secret := rand.Text()
	environment := append(os.Environ(), "MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID=container-test",
		"MARIE_PLUGIN_DAEMON_SIGNING_SECRET="+secret)
	id := containerCommand(t, environment, "create", "--pull=never",
		"--user", "10001:10001", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
		"--memory=512m", "--cpus=1", "--pids-limit=128",
		"--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777,size=128m",
		"--tmpfs", "/mnt/data/marie-ai:rw,nosuid,nodev,uid=10001,gid=10001,mode=0700,size=128m",
		"--publish", "127.0.0.1::8099",
		"--env", "MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID", "--env", "MARIE_PLUGIN_DAEMON_SIGNING_SECRET",
		"--env", "UV_OFFLINE=1", "--env", "UV_PYTHON_DOWNLOADS=never", image)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		if err := exec.CommandContext(ctx, "docker", "rm", "--force", id).Run(); err != nil {
			t.Errorf("remove owned test container: %v", err)
		}
	})
	containerCommand(t, nil, "start", id)
	address := containerCommand(t, nil, "port", id, "8099/tcp")
	host, _, err := net.SplitHostPort(address)
	if err != nil || host != "127.0.0.1" {
		t.Fatalf("expected a single loopback port binding, got %q", address)
	}
	client := &http.Client{Timeout: 30 * time.Second, Transport: &http.Transport{Proxy: nil}}
	t.Cleanup(client.CloseIdleConnections)
	baseURL := "http://" + address
	deadline := time.Now().Add(30 * time.Second)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/health", nil)
		if err != nil {
			cancel()
			t.Fatal(err)
		}
		response, err := client.Do(request)
		ready := err == nil && response.StatusCode == http.StatusOK
		if response != nil {
			response.Body.Close()
		}
		cancel()
		if ready {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("container did not become healthy within 30 seconds")
		}
		time.Sleep(100 * time.Millisecond)
	}
	archive := containerPluginArchive(t)
	_, digest := archiveIdentity(t, archive)
	envelope := func(payload map[string]any) map[string]any {
		value := runtimeEnvelope(t, map[string]any{"packageDigest": digest, "actionType": "tool", "payload": payload})
		value["signature"] = map[string]any{
			"keyId": "container-test", "algorithm": "hmac-sha256", "value": signEnvelope(value, secret),
		}
		return value
	}
	request := func(t *testing.T, method, path string, data []byte, header map[string]any) (int, http.Header, []byte) {
		t.Helper()
		req, err := http.NewRequest(method, baseURL+path, bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		if header != nil {
			req.Header.Set("X-Marie-Envelope", envelopeHeader(t, header))
		}
		response, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		if err != nil {
			t.Fatal(err)
		}
		return response.StatusCode, response.Header, body
	}
	t.Run("authentication", func(t *testing.T) {
		code, _, _ := request(t, http.MethodPost, "/v1/plugins/install", archive, nil)
		if code != http.StatusUnauthorized {
			t.Fatalf("unsigned install returned %d", code)
		}
		invalid := envelope(nil)
		invalid["signature"].(map[string]any)["value"] = "invalid"
		code, _, _ = request(t, http.MethodPost, "/v1/plugins/install", archive, invalid)
		if code != http.StatusUnauthorized {
			t.Fatalf("invalid signature returned %d", code)
		}
	})
	t.Run("signed_install_invoke_redeploy_remove", func(t *testing.T) {
		var workingDir string
		for attempt := 0; attempt < 2; attempt++ {
			code, _, body := request(t, http.MethodPost, "/v1/plugins/install", archive, envelope(nil))
			if code != http.StatusOK {
				t.Fatalf("signed installation returned %d: %s", code, body)
			}
			var installed struct {
				State   string `json:"state"`
				Install struct {
					WorkingDir string `json:"workingDir"`
				} `json:"install"`
			}
			if err := json.Unmarshal(body, &installed); err != nil {
				t.Fatal(err)
			}
			if installed.State != "ready" || !strings.HasPrefix(installed.Install.WorkingDir, "/mnt/data/marie-ai/plugin-runtime/org1__ws1/") {
				t.Fatalf("unexpected installed state/location: %s", body)
			}
			if attempt == 1 && workingDir != installed.Install.WorkingDir {
				t.Fatal("identical redeploy changed the active environment")
			}
			workingDir = installed.Install.WorkingDir
			invoke, err := json.Marshal(envelope(map[string]any{"message": "container-roundtrip"}))
			if err != nil {
				t.Fatal(err)
			}
			code, headers, body := request(t, http.MethodPost, "/v1/dispatch/invoke", invoke, nil)
			if code != http.StatusOK || headers.Get("Content-Type") != "text/event-stream" {
				t.Fatalf("invoke returned %d: %s", code, body)
			}
			var stream, end bool
			for _, line := range strings.Split(string(body), "\n") {
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				var frame struct {
					Type string `json:"type"`
					Data struct {
						Echo       map[string]any `json:"echo"`
						Dependency string         `json:"dependency"`
						UID        int            `json:"uid"`
						Python     string         `json:"python"`
					} `json:"data"`
				}
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &frame); err != nil {
					t.Fatal(err)
				}
				if frame.Type == "stream" {
					stream = frame.Data.Echo["message"] == "container-roundtrip" && frame.Data.Dependency == "offline-wheel" &&
						frame.Data.UID == 10001 && strings.HasPrefix(frame.Data.Python, workingDir+"/.venv/")
				}
				if frame.Type == "end" {
					end = true
				}
			}
			if !stream || !end {
				t.Fatalf("missing verified Python result/end: %s", body)
			}
		}
		code, _, body := request(t, http.MethodDelete, "/v1/plugins/"+fixturePackageRef, nil, envelope(nil))
		if code != http.StatusOK {
			t.Fatalf("remove returned %d: %s", code, body)
		}
		invoke, _ := json.Marshal(envelope(nil))
		code, _, _ = request(t, http.MethodPost, "/v1/dispatch/invoke", invoke, nil)
		if code != http.StatusNotFound {
			t.Fatalf("removed plugin invoke returned %d", code)
		}
	})
}

func containerCommand(t *testing.T, environment []string, args ...string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "docker", args...)
	command.Env = environment
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("docker %s failed: %v: %s", args[0], err, output)
	}
	return strings.TrimSpace(string(output))
}

func containerPluginArchive(t *testing.T) []byte {
	t.Helper()
	wheel := zipArchive(t, map[string][]byte{
		"fixture_dep/__init__.py":              []byte("VALUE = 'offline-wheel'\n"),
		"fixture_dep-1.0.0.dist-info/METADATA": []byte("Metadata-Version: 2.1\nName: fixture-dep\nVersion: 1.0.0\n"),
		"fixture_dep-1.0.0.dist-info/WHEEL":    []byte("Wheel-Version: 1.0\nGenerator: marie-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
		"fixture_dep-1.0.0.dist-info/RECORD":   []byte("fixture_dep/__init__.py,,\nfixture_dep-1.0.0.dist-info/METADATA,,\nfixture_dep-1.0.0.dist-info/WHEEL,,\nfixture_dep-1.0.0.dist-info/RECORD,,\n"),
	})
	return zipArchive(t, map[string][]byte{
		"marie-extension.yaml":                      []byte("apiVersion: marie.ai/v1alpha1\nkind: ExtensionPackage\nmetadata:\n  id: ext.test.fixture-echo\n  author: marie\n  name: fixture-echo\n  version: 0.0.1\nruntime:\n  type: python_source\n  language: python\n  version: '3.12'\n  entrypoint: main\n"),
		"requirements.txt":                          []byte("./wheels/fixture_dep-1.0.0-py3-none-any.whl\n"),
		"wheels/fixture_dep-1.0.0-py3-none-any.whl": wheel,
		"main.py": []byte(`import os
import sys
from fixture_dep import VALUE
from marie_plugins.runtime import run, session_frame

def dispatch(request):
    sid = request["session_id"]
    return [session_frame(sid, "stream", {"echo": request["data"], "dependency": VALUE,
            "uid": os.getuid(), "python": sys.executable}), session_frame(sid, "end", {})]

run(dispatch)
`),
	})
}
