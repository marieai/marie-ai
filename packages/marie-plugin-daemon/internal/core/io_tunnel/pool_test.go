package io_tunnel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	backwards_invocation "github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel/backwards_invocation"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/local_runtime"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/plugin_manager"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/decoder"
)

const testTenant = "org1__ws1"

const multiplexPluginScript = `import json
import sys
import threading
import time

lock = threading.Lock()
pending = []

def emit(session_id, kind, data):
    with lock:
        print(json.dumps({"session_id": session_id, "event": "session",
                          "data": {"type": kind, "data": data}}), flush=True)

def heartbeats():
    while True:
        with lock:
            print(json.dumps({"session_id": "", "event": "heartbeat", "data": None}), flush=True)
        time.sleep(1)

threading.Thread(target=heartbeats, daemon=True).start()
for line in sys.stdin:
    request = json.loads(line)
    sid = request["session_id"]
    payload = request["data"]
    if payload.get("hold"):
        pending.append(sid)
        emit(sid, "stream", {"waiting": True})
        for index in range(payload.get("burst", 0)):
            emit(sid, "stream", {"index": index})
    else:
        for waiting in pending:
            emit(waiting, "stream", {"released": True})
            emit(waiting, "end", {})
        pending.clear()
        emit(sid, "end", {})
`

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

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{"ping": true}, 10*time.Second)
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

func TestInvokeChecksRunningWorkerDigest(t *testing.T) {
	pool, install := newDeployedPool(t, fixtureZip(t))
	// The selected on-disk version can change before its worker is ready.
	other, err := pool.manager.Install(testTenant, scriptZip(t, install.PackageRef, slowPluginScript))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, other.Digest, map[string]any{}, time.Second); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("invocation ignored running worker digest: %v", err)
	}
	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{}, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	result := collectFrames(t, frames, 2*time.Second)
	if len(result) != 2 || result[0].Type != FrameStream || result[1].Type != FrameEnd {
		t.Fatalf("correct worker version did not execute: %+v", result)
	}
}

func TestReplacementKeepsOldWorkerUntilReady(t *testing.T) {
	for _, remove := range []bool{false, true} {
		t.Run(fmt.Sprintf("remove=%t", remove), func(t *testing.T) {
			pool, old := newDeployedPool(t, fixtureZip(t))
			gate := t.TempDir()
			started, release := filepath.Join(gate, "started"), filepath.Join(gate, "release")
			script := fmt.Sprintf("from pathlib import Path\nimport time\nPath(%q).touch()\nwhile not Path(%q).exists(): time.sleep(0.01)\n", started, release) + multiplexPluginScript
			archive := scriptZip(t, old.PackageRef, script)
			_, nextDigest, err := plugin_manager.Inspect(archive)
			if err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			finished := make(chan struct{})
			go func() {
				defer close(finished)
				_, err := pool.Install(context.Background(), testTenant, archive)
				done <- err
			}()
			defer func() {
				os.WriteFile(release, nil, 0o644)
				select {
				case <-finished:
				case <-time.After(10 * time.Second):
					t.Error("replacement did not finish")
				}
			}()
			deadline := time.Now().Add(10 * time.Second)
			for {
				if _, err := os.Stat(started); err == nil {
					break
				}
				if time.Now().After(deadline) {
					t.Fatal("replacement never reached startup gate")
				}
				time.Sleep(10 * time.Millisecond)
			}
			if current, ok := pool.manager.Get(testTenant, old.PackageRef); !ok || current.Digest != old.Digest {
				t.Fatalf("unready replacement changed active install: %+v", current)
			}
			frames, err := pool.Invoke(context.Background(), testTenant, old.PackageRef, old.Digest, map[string]any{}, time.Second)
			if err != nil {
				t.Fatalf("old worker unavailable during replacement: %v", err)
			}
			if got := collectFrames(t, frames, 2*time.Second); len(got) != 2 || got[0].Type != FrameStream {
				t.Fatalf("old worker did not serve: %+v", got)
			}
			if _, err := pool.Invoke(context.Background(), testTenant, old.PackageRef, nextDigest, map[string]any{}, time.Second); !errors.Is(err, ErrDigestMismatch) {
				t.Fatalf("candidate digest reached old worker: %v", err)
			}
			if remove {
				if err := pool.Remove(testTenant, old.PackageRef); err != nil {
					t.Fatal(err)
				}
			}
			if err := os.WriteFile(release, nil, 0o644); err != nil {
				t.Fatal(err)
			}
			var deployErr error
			select {
			case deployErr = <-done:
			case <-time.After(10 * time.Second):
				t.Fatal("replacement stuck")
			}
			if remove {
				if !errors.Is(deployErr, ErrDeployCancelled) {
					t.Fatalf("removed package resurrected: %v", deployErr)
				}
				if _, ok := pool.manager.Get(testTenant, old.PackageRef); ok {
					t.Fatal("removed package was reactivated")
				}
				return
			}
			if deployErr != nil {
				t.Fatal(deployErr)
			}
			if _, err := os.Stat(old.WorkingDir); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("retired files not cleaned up: %v", err)
			}
			if _, err := pool.Invoke(context.Background(), testTenant, old.PackageRef, old.Digest, map[string]any{}, time.Second); !errors.Is(err, ErrDigestMismatch) {
				t.Fatalf("stale digest reached replacement: %v", err)
			}
			frames, err = pool.Invoke(context.Background(), testTenant, old.PackageRef, nextDigest, map[string]any{}, time.Second)
			if err != nil {
				t.Fatal(err)
			}
			if got := collectFrames(t, frames, 2*time.Second); len(got) != 1 || got[0].Type != FrameEnd {
				t.Fatalf("replacement did not serve: %+v", got)
			}
		})
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
	if _, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{}, time.Second); !errors.Is(err, ErrNotRunning) {
		t.Fatalf("expected instance_not_running, got %v", err)
	}
}

func TestInvokeTimeoutEmitsErrorFrame(t *testing.T) {
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.slow", slowPluginScript))

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{}, 300*time.Millisecond)
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

func TestInvocationDeadlineIncludesBlockedStdin(t *testing.T) {
	for _, limit := range []string{"timeout", "context_deadline", "cancellation"} {
		t.Run(limit, func(t *testing.T) {
			script := strings.Replace(slowPluginScript, "for line in sys.stdin:\n    pass", "threading.Event().wait()", 1)
			pool, install := newDeployedPool(t, scriptZip(t, "ext.test.blocked", script))
			ctx := context.Background()
			timeout := 100 * time.Millisecond
			if limit == "context_deadline" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, timeout)
				defer cancel()
				timeout = 10 * time.Second
			} else if limit == "cancellation" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				defer cancel()
				timer := time.AfterFunc(timeout, cancel)
				defer timer.Stop()
				timeout = 10 * time.Second
			}
			done := make(chan error, 1)
			go func() {
				frames, err := pool.Invoke(ctx, testTenant, install.PackageRef, install.Digest, map[string]any{"data": strings.Repeat("x", 2<<20)}, timeout)
				if err == nil {
					var result []Frame
					for frame := range frames {
						result = append(result, frame)
					}
					if limit == "timeout" && (len(result) != 1 || result[0].Type != FrameError || !strings.Contains(string(result[0].Data), `"retryable":false`)) {
						err = fmt.Errorf("expected conservative timeout error: %+v", result)
					}
				} else if limit != "timeout" && errors.Is(err, ctx.Err()) {
					err = nil
				}
				done <- err
			}()
			select {
			case err := <-done:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(time.Second):
				t.Fatal("invocation remained blocked past its deadline writing stdin")
			}
			if pool.ReadyCount() != 1 {
				t.Fatal("request timeout stopped shared plugin")
			}
			baseline := runtime.NumGoroutine()
			for range 20 {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
				_, err := pool.Invoke(ctx, testTenant, install.PackageRef, install.Digest, map[string]any{}, time.Second)
				cancel()
				if err != nil && !errors.Is(err, context.DeadlineExceeded) {
					t.Fatal(err)
				}
			}
			if growth := runtime.NumGoroutine() - baseline; growth > 2 {
				t.Fatalf("expired admissions left blocked goroutines: +%d", growth)
			}
		})
	}
}

func TestTimedOutWritePreservesFramingAndOtherSessions(t *testing.T) {
	script := "import os\n" + strings.Replace(multiplexPluginScript, "    request = json.loads(line)\n", `    request = json.loads(line)
    if request["data"].get("pause"):
        emit(request["session_id"], "stream", {"paused": True})
        while not os.path.exists("resume"):
            time.sleep(0.01)
        emit(request["session_id"], "end", {})
        continue
    if request["data"].get("late"):
        with open("received", "a") as received:
            received.write(request["data"]["late"] + "\n")
`, 1)
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.resume-reader", script))
	survivor, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{"hold": true}, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case frame := <-survivor:
		if !strings.Contains(string(frame.Data), "waiting") {
			t.Fatalf("survivor not started: %+v", frame)
		}
	case <-time.After(time.Second):
		t.Fatal("survivor did not start")
	}
	paused, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{"pause": true}, 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case frame := <-paused:
		if !strings.Contains(string(frame.Data), "paused") {
			t.Fatalf("reader not paused: %+v", frame)
		}
	case <-time.After(time.Second):
		t.Fatal("reader did not pause")
	}
	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{"data": strings.Repeat("x", 2<<20), "late": "started"}, 100*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if got := collectFrames(t, frames, time.Second); len(got) != 1 || got[0].Type != FrameError {
		t.Fatalf("write did not time out: %+v", got)
	}
	for range 10 {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
		_, err := pool.Invoke(ctx, testTenant, install.PackageRef, install.Digest, map[string]any{"late": "expired"}, time.Second)
		cancel()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("expired admission returned %v", err)
		}
	}
	if err := os.WriteFile(filepath.Join(install.WorkingDir, "resume"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	frames, err = pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{}, 3*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if got := collectFrames(t, frames, 4*time.Second); len(got) != 1 || got[0].Type != FrameEnd {
		t.Fatalf("follow-up lost JSON framing: %+v", got)
	}
	if got := collectFrames(t, survivor, time.Second); len(got) != 2 || got[0].Type != FrameStream || got[1].Type != FrameEnd {
		t.Fatalf("unrelated session interrupted: %+v", got)
	}
	if got := collectFrames(t, paused, time.Second); len(got) != 1 || got[0].Type != FrameEnd {
		t.Fatalf("pause session interrupted: %+v", got)
	}
	received, err := os.ReadFile(filepath.Join(install.WorkingDir, "received"))
	if err != nil {
		t.Fatal(err)
	}
	if string(received) != "started\n" {
		t.Fatalf("expired admissions executed: %q", received)
	}
}

func TestInvocationDeadlineIncludesBackwardsWrite(t *testing.T) {
	script := strings.Replace(slowPluginScript, "for line in sys.stdin:\n    pass", `for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({"session_id": request["session_id"], "event": "session", "data": {
        "type": "invoke", "data": {"type": "storage", "backwards_request_id": "read-large",
        "request": {"opt": "get", "key": "large"}}}}), flush=True)
    threading.Event().wait()`, 1)
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.block-backwards", script))
	if err := pool.store.Set(testTenant, install.PackageRef, "large", []byte(strings.Repeat("x", 512<<10))); err != nil {
		t.Fatal(err)
	}
	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{}, 100*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	got := collectFrames(t, frames, time.Second)
	if len(got) == 0 || got[len(got)-1].Type != FrameError || !strings.Contains(string(got[len(got)-1].Data), "timeout") {
		t.Fatalf("backwards write did not time out: %+v", got)
	}
	if pool.ReadyCount() != 1 {
		t.Fatal("backwards timeout stopped shared instance")
	}
}

func TestInvocationTimeoutDoesNotRestartAfterWrite(t *testing.T) {
	script := "import os\n" + strings.Replace(slowPluginScript, "for line in sys.stdin:", "while not os.path.exists('resume'):\n    time.sleep(0.01)\nfor line in sys.stdin:", 1)
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.remaining-deadline", script))
	started := time.Now()
	release := time.AfterFunc(300*time.Millisecond, func() { os.WriteFile(filepath.Join(install.WorkingDir, "resume"), nil, 0o644) })
	defer release.Stop()
	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{"data": strings.Repeat("x", 2<<20)}, 500*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	got := collectFrames(t, frames, time.Second)
	if len(got) != 1 || got[0].Type != FrameError {
		t.Fatalf("expected invocation timeout: %+v", got)
	}
	if elapsed := time.Since(started); elapsed > 650*time.Millisecond {
		t.Fatalf("write reset timeout budget: %s", elapsed)
	}
}

func TestInvokeClientCancelClosesFrames(t *testing.T) {
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.slow", slowPluginScript))

	ctx, cancel := context.WithCancel(context.Background())
	frames, err := pool.Invoke(ctx, testTenant, install.PackageRef, install.Digest, map[string]any{}, 30*time.Second)
	if err != nil {
		t.Fatalf("invoke failed: %v", err)
	}
	cancel()

	collected := collectFrames(t, frames, 2*time.Second)
	if len(collected) != 0 {
		t.Fatalf("expected no frames after cancel, got %+v", collected)
	}
}

func TestInvokeAbortPreservesOtherSessions(t *testing.T) {
	for _, abort := range []string{"timeout", "cancel"} {
		for _, burst := range []int{0, frameBufferSize + 4} {
			t.Run(fmt.Sprintf("%s/burst=%d", abort, burst), func(t *testing.T) {
				pool, install := newDeployedPool(t, scriptZip(t, "ext.test.multiplex", multiplexPluginScript))
				survivor, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{"hold": true}, 10*time.Second)
				if err != nil {
					t.Fatal(err)
				}
				select {
				case frame := <-survivor:
					if frame.Type != FrameStream || !strings.Contains(string(frame.Data), `"waiting": true`) {
						t.Fatalf("survivor did not start: %+v", frame)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("survivor did not start")
				}

				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				timeout := 200 * time.Millisecond
				if abort == "cancel" {
					timeout = 10 * time.Second
				}
				frames, err := pool.Invoke(ctx, testTenant, install.PackageRef, install.Digest, map[string]any{"hold": true, "burst": burst}, timeout)
				if err != nil {
					t.Fatal(err)
				}
				select {
				case frame := <-frames:
					if frame.Type != FrameStream {
						t.Fatalf("aborted session did not start: %+v", frame)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("aborted session did not start")
				}
				if abort == "cancel" {
					cancel()
				}
				// Leave a burst unread until the deadline to exercise blocked emit.
				if burst > 0 && abort == "timeout" {
					time.Sleep(2 * timeout)
				}
				aborted := collectFrames(t, frames, 5*time.Second)
				if abort == "timeout" && burst == 0 {
					if len(aborted) != 1 || aborted[0].Type != FrameError {
						t.Fatalf("expected timeout error: %+v", aborted)
					}
					var data struct {
						Retryable bool `json:"retryable"`
					}
					if err := json.Unmarshal(aborted[0].Data, &data); err != nil {
						t.Fatal(err)
					}
					if data.Retryable {
						t.Error("unknown plugin outcome must not invite automatic retry")
					}
				}
				release, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{}, 5*time.Second)
				if err != nil {
					t.Fatalf("request abort killed shared plugin: %v", err)
				}
				if got := collectFrames(t, release, 5*time.Second); len(got) != 1 || got[0].Type != FrameEnd {
					t.Fatalf("follow-up request failed: %+v", got)
				}
				got := collectFrames(t, survivor, 5*time.Second)
				if len(got) != 2 || got[0].Type != FrameStream || !strings.Contains(string(got[0].Data), `"released": true`) || got[1].Type != FrameEnd {
					t.Fatalf("other session interrupted: %+v", got)
				}
			})
		}
	}
}

func TestSlowReaderConsumerExitsAtTimeout(t *testing.T) {
	pool, install := newDeployedPool(t, scriptZip(t, "ext.test.flood", floodPluginScript))

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{}, 300*time.Millisecond)
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

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{"emit_invoke": true}, 10*time.Second)
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

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest,
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
	frames, err = pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest,
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

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest,
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

	frames, err := pool.Invoke(context.Background(), testTenant, install.PackageRef, install.Digest, map[string]any{}, 30*time.Second)
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
