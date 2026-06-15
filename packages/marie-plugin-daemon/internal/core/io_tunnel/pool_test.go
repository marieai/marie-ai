package io_tunnel

import (
	"context"
	"errors"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	backwards_invocation "github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel/backwards_invocation"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/local_runtime"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/plugin_manager"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/decoder"
)

const testTenant = "org1__ws1"

const slowPluginScript = `import json
import sys
import threading
import time

def heartbeats():
    while True:
        sys.stdout.write(json.dumps({"session_id": "", "event": "heartbeat", "data": None}) + "\n")
        sys.stdout.flush()
        time.sleep(1)

threading.Thread(target=heartbeats, daemon=True).start()
for line in sys.stdin:
    pass
`

const floodPluginScript = `import json
import sys
import threading
import time

def heartbeats():
    while True:
        sys.stdout.write(json.dumps({"session_id": "", "event": "heartbeat", "data": None}) + "\n")
        sys.stdout.flush()
        time.sleep(1)

threading.Thread(target=heartbeats, daemon=True).start()
for line in sys.stdin:
    request = json.loads(line)
    session_id = request["session_id"]
    for index in range(2000):
        sys.stdout.write(json.dumps({
            "session_id": session_id,
            "event": "session",
            "data": {"type": "stream", "data": {"index": index}},
        }) + "\n")
    sys.stdout.flush()
    open("flood-done", "w").write("done")
    sys.stdout.write(json.dumps({
        "session_id": session_id,
        "event": "session",
        "data": {"type": "end", "data": {}},
    }) + "\n")
    sys.stdout.flush()
`

func requireUV(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not installed")
	}
}

func fixtureZip(t *testing.T) []byte {
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
	return zipFromFiles(t, files)
}

func scriptZip(t *testing.T, packageRef string, script string) []byte {
	t.Helper()
	manifest := "apiVersion: marie.ai/v1alpha1\nkind: ExtensionPackage\nmetadata:\n  id: " + packageRef + "\n  author: marie\n  name: test\n  version: 0.0.1\nruntime:\n  type: python_source\n  language: python\n  version: \"3.12\"\n  entrypoint: main\n"
	return zipFromFiles(t, map[string][]byte{
		"marie-extension.yaml": []byte(manifest),
		"main.py":              []byte(script),
	})
}

func zipFromFiles(t *testing.T, files map[string][]byte) []byte {
	t.Helper()
	zipBytes, err := decoder.ZipFixture(files)
	if err != nil {
		t.Fatalf("zip fixture: %v", err)
	}
	return zipBytes
}

func newDeployedPool(t *testing.T, archive []byte) (*Pool, plugin_manager.Install) {
	t.Helper()
	requireUV(t)

	root := t.TempDir()
	manager := plugin_manager.NewManager(root)
	pool := NewPool(manager, backwards_invocation.NewStorage(root), io.Discard)
	t.Cleanup(pool.Shutdown)

	install, err := manager.Install(testTenant, archive)
	if err != nil {
		t.Fatalf("install failed: %v", err)
	}
	deployed, err := pool.Deploy(context.Background(), testTenant, install.PackageRef)
	if err != nil {
		t.Fatalf("deploy failed: %v", err)
	}
	return pool, deployed
}

func collectFrames(t *testing.T, frames <-chan Frame, within time.Duration) []Frame {
	t.Helper()
	collected := []Frame{}
	deadline := time.After(within)
	for {
		select {
		case frame, ok := <-frames:
			if !ok {
				return collected
			}
			collected = append(collected, frame)
		case <-deadline:
			t.Fatalf("frames channel not closed within %s, got %+v", within, collected)
		}
	}
}

func TestDeployIdempotent(t *testing.T) {
	pool, install := newDeployedPool(t, fixtureZip(t))

	if install.State != plugin_manager.StateReady {
		t.Fatalf("expected ready after deploy, got %q", install.State)
	}
	if state := pool.InstanceState(testTenant, install.PackageRef); state != local_runtime.InstanceStateReady {
		t.Fatalf("expected ready instance, got %q", state)
	}
	if count := pool.ReadyCount(); count != 1 {
		t.Fatalf("expected 1 ready instance, got %d", count)
	}

	again, err := pool.Deploy(context.Background(), testTenant, install.PackageRef)
	if err != nil {
		t.Fatalf("second deploy failed: %v", err)
	}
	if again.State != plugin_manager.StateReady {
		t.Fatalf("expected ready after redeploy, got %q", again.State)
	}
	if count := pool.ReadyCount(); count != 1 {
		t.Fatalf("expected single instance after redeploy, got %d", count)
	}
}

func TestDeployUnknownPackageFails(t *testing.T) {
	pool := NewPool(plugin_manager.NewManager(t.TempDir()), backwards_invocation.NewStorage(t.TempDir()), io.Discard)
	t.Cleanup(pool.Shutdown)

	if _, err := pool.Deploy(context.Background(), testTenant, "ext.test.missing"); !errors.Is(err, ErrNotInstalled) {
		t.Fatalf("expected plugin_not_installed, got %v", err)
	}
}

func TestInvokeStreamsFramesThenEnd(t *testing.T) {
	pool, install := newDeployedPool(t, fixtureZip(t))

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, map[string]any{"ping": true}, 10*time.Second)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}
	collected := collectFrames(t, frames, 10*time.Second)
	if len(collected) != 2 {
		t.Fatalf("expected stream+end, got %+v", collected)
	}
	if collected[0].Type != FrameStream {
		t.Fatalf("expected stream frame first, got %+v", collected[0])
	}
	data := string(collected[0].Data)
	if !strings.Contains(data, `"echo"`) || !strings.Contains(data, `"event": "request"`) && !strings.Contains(data, `"event":"request"`) {
		t.Fatalf("unexpected stream data: %s", data)
	}
	if collected[1].Type != FrameEnd {
		t.Fatalf("expected end frame, got %+v", collected[1])
	}
}

func TestInvokeAfterStopFails(t *testing.T) {
	pool, install := newDeployedPool(t, fixtureZip(t))

	if err := pool.Stop(testTenant, install.PackageRef); err != nil {
		t.Fatalf("stop failed: %v", err)
	}
	if state := pool.InstanceState(testTenant, install.PackageRef); state != StateAbsent {
		t.Fatalf("expected absent after stop, got %q", state)
	}
	if _, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, map[string]any{}, time.Second); !errors.Is(err, ErrNotRunning) {
		t.Fatalf("expected instance_not_running, got %v", err)
	}
}

func TestInvokeTimeoutEmitsErrorFrame(t *testing.T) {
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.slow", slowPluginScript))

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, map[string]any{}, 300*time.Millisecond)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}
	collected := collectFrames(t, frames, 10*time.Second)
	if len(collected) != 1 || collected[0].Type != FrameError {
		t.Fatalf("expected single error frame, got %+v", collected)
	}
	if !strings.Contains(string(collected[0].Data), "timeout") {
		t.Fatalf("expected timeout message, got %s", collected[0].Data)
	}
}

func TestInvokeClientCancelClosesFrames(t *testing.T) {
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.slow", slowPluginScript))

	ctx, cancel := context.WithCancel(context.Background())
	frames, err := pool.Invoke(ctx, testTenant, install.PackageRef, map[string]any{}, 30*time.Second)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}
	cancel()

	collected := collectFrames(t, frames, 2*time.Second)
	if len(collected) != 0 {
		t.Fatalf("expected no frames after cancel, got %+v", collected)
	}
}

func TestSlowReaderConsumerExitsAtTimeout(t *testing.T) {
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.flood", floodPluginScript))

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, map[string]any{}, 300*time.Millisecond)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}

	// Read nothing until well past the timeout: the consumer must abort the
	// blocked emit at the deadline instead of waiting for a reader.
	time.Sleep(1200 * time.Millisecond)

	collected := collectFrames(t, frames, 2*time.Second)
	if len(collected) > frameBufferSize {
		t.Fatalf("consumer kept streaming past the timeout: %d frames", len(collected))
	}
}

func TestBackwardsInvocationRejected(t *testing.T) {
	pool, install := newDeployedPool(t, fixtureZip(t))

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, map[string]any{"emit_invoke": true}, 10*time.Second)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}
	collected := collectFrames(t, frames, 10*time.Second)
	if len(collected) != 3 {
		t.Fatalf("expected log+stream+end, got %+v", collected)
	}
	if collected[0].Type != FrameLog || !strings.Contains(string(collected[0].Data), "r1") {
		t.Fatalf("expected rejection log frame with request id, got %+v", collected[0])
	}
	stream := string(collected[1].Data)
	if collected[1].Type != FrameStream || !strings.Contains(stream, "backwards invocation not supported") || !strings.Contains(stream, "r1") {
		t.Fatalf("expected echoed backwards_response, got %+v", collected[1])
	}
	if collected[2].Type != FrameEnd {
		t.Fatalf("expected end frame, got %+v", collected[2])
	}
}

func storageChunk(opt, key, value string) backwardsInvokeChunk {
	chunk := backwardsInvokeChunk{Type: "storage", BackwardsRequestID: "r1"}
	chunk.Request.Opt = opt
	chunk.Request.Key = key
	chunk.Request.Value = value
	return chunk
}

func TestExecuteStorageOpPerOpResponses(t *testing.T) {
	pool := NewPool(nil, backwards_invocation.NewStorage(t.TempDir()), nil)
	if _, err := pool.executeStorageOp(testTenant, "ext.test.kv", storageChunk("set", "present", "7631")); err != nil {
		t.Fatalf("seed set failed: %v", err)
	}

	tests := []struct {
		name    string
		chunk   backwardsInvokeChunk
		want    map[string]any
		wantErr string
	}{
		{
			name:  "exist true",
			chunk: storageChunk("exist", "present", ""),
			want:  map[string]any{"data": true, "exist_num": 1},
		},
		{
			name:  "exist false",
			chunk: storageChunk("exist", "missing", ""),
			want:  map[string]any{"data": false, "exist_num": 0},
		},
		{
			name:  "del existing key",
			chunk: storageChunk("del", "present", ""),
			want:  map[string]any{"data": "ok", "deleted_num": 1},
		},
		{
			name:  "del missing key",
			chunk: storageChunk("del", "present", ""),
			want:  map[string]any{"data": "ok", "deleted_num": 0},
		},
		{
			name:    "unsupported opt",
			chunk:   storageChunk("purge", "present", ""),
			wantErr: `unsupported storage opt: "purge"`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result, err := pool.executeStorageOp(testTenant, "ext.test.kv", test.chunk)
			if test.wantErr != "" {
				if err == nil || err.Error() != test.wantErr {
					t.Fatalf("expected error %q, got %v (result %v)", test.wantErr, err, result)
				}
				event := backwardsErrorEvent(test.chunk.BackwardsRequestID, err.Error())
				want := map[string]any{
					"backwards_request_id": "r1",
					"event":                "error",
					"message":              test.wantErr,
					"data":                 nil,
				}
				if !reflect.DeepEqual(event, want) {
					t.Fatalf("error event mismatch:\n got %#v\nwant %#v", event, want)
				}
				return
			}
			if err != nil {
				t.Fatalf("executeStorageOp failed: %v", err)
			}
			if !reflect.DeepEqual(result, test.want) {
				t.Fatalf("response mismatch:\n got %#v\nwant %#v", result, test.want)
			}
		})
	}
}

func finalStreamFrame(t *testing.T, collected []Frame) Frame {
	t.Helper()
	if len(collected) < 2 {
		t.Fatalf("expected at least stream+end, got %+v", collected)
	}
	last := collected[len(collected)-1]
	if last.Type != FrameEnd {
		t.Fatalf("expected trailing end frame, got %+v", collected)
	}
	stream := collected[len(collected)-2]
	if stream.Type != FrameStream {
		t.Fatalf("expected stream frame before end, got %+v", collected)
	}
	return stream
}

func TestStorageRoundTripThroughPlugin(t *testing.T) {
	pool, install := newDeployedPool(t, fixtureZip(t))

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef,
		map[string]any{"storage_roundtrip": true, "key": "checkpoint", "value": "v1"}, 20*time.Second)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}
	collected := collectFrames(t, frames, 20*time.Second)

	logs := 0
	for _, frame := range collected {
		if frame.Type == FrameLog {
			logs++
			data := string(frame.Data)
			if strings.Contains(data, "v1") || strings.Contains(data, "checkpoint") {
				t.Fatalf("log frame leaked key or value: %s", data)
			}
			if !strings.Contains(data, "storage backwards invocation") {
				t.Fatalf("expected storage op log frame, got %s", data)
			}
		}
	}
	if logs != 2 {
		t.Fatalf("expected log frames for set+get, got %+v", collected)
	}
	stream := finalStreamFrame(t, collected)
	if !strings.Contains(string(stream.Data), `"storage_get"`) || !strings.Contains(string(stream.Data), `"v1"`) {
		t.Fatalf("expected GET to return v1, got %s", stream.Data)
	}

	// A fresh instance must see the same value: stop, redeploy, GET only.
	if err := pool.Stop(testTenant, install.PackageRef); err != nil {
		t.Fatalf("stop failed: %v", err)
	}
	if _, err := pool.Deploy(context.Background(), testTenant, install.PackageRef); err != nil {
		t.Fatalf("redeploy failed: %v", err)
	}
	frames, err = pool.Invoke(context.Background(), testTenant, install.PackageRef,
		map[string]any{"storage_roundtrip_get_only": true, "key": "checkpoint"}, 20*time.Second)
	if err != nil {
		t.Fatalf("invoke after restart failed: %v", err)
	}
	stream = finalStreamFrame(t, collectFrames(t, frames, 20*time.Second))
	if !strings.Contains(string(stream.Data), `"v1"`) {
		t.Fatalf("expected persisted value after restart, got %s", stream.Data)
	}
}

func TestStorageGetMissingKeyReturnsErrorEvent(t *testing.T) {
	pool, install := newDeployedPool(t, fixtureZip(t))

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef,
		map[string]any{"storage_roundtrip_get_only": true, "key": "never-set"}, 20*time.Second)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}
	stream := finalStreamFrame(t, collectFrames(t, frames, 20*time.Second))
	if !strings.Contains(string(stream.Data), `"storage_error"`) || !strings.Contains(string(stream.Data), "load data failed") {
		t.Fatalf("expected upstream-shaped storage error, got %s", stream.Data)
	}
}

func TestInvokeOverflowEmitsErrorFrame(t *testing.T) {
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.flood", floodPluginScript))

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, map[string]any{}, 30*time.Second)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}

	// Do not consume frames until the plugin has finished flooding, forcing
	// the per-session buffer to overflow.
	doneFile := filepath.Join(install.WorkingDir, "flood-done")
	deadline := time.Now().Add(20 * time.Second)
	for {
		if _, err := os.Stat(doneFile); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("flood plugin never finished")
		}
		time.Sleep(20 * time.Millisecond)
	}

	collected := collectFrames(t, frames, 10*time.Second)
	last := collected[len(collected)-1]
	if last.Type != FrameError || !strings.Contains(string(last.Data), "overflow") {
		t.Fatalf("expected trailing overflow error frame, got %+v", last)
	}
}
