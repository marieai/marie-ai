package local_runtime

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	plugin_entities "github.com/marieai/marie-ai/packages/marie-plugin-daemon/pkg/entities/plugin_entities"
)

const heartbeatingPlugin = `import json
import os
import sys
import threading
import time

open("pid", "w").write(str(os.getpid()))

def heartbeats():
    while True:
        sys.stdout.write(json.dumps({"session_id": "", "event": "heartbeat", "data": None}) + "\n")
        sys.stdout.flush()
        time.sleep(1)

threading.Thread(target=heartbeats, daemon=True).start()
time.sleep(120)
`

const silentPlugin = `import os
import time

open("pid", "w").write(str(os.getpid()))
time.sleep(120)
`

const singleHeartbeatPlugin = `import json
import os
import sys
import time

open("pid", "w").write(str(os.getpid()))
sys.stdout.write(json.dumps({"session_id": "", "event": "heartbeat", "data": None}) + "\n")
sys.stdout.flush()
time.sleep(120)
`

func fixturePluginDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(filepath.Join("..", "..", "..", "testdata", "fixture-plugin"))
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func writePlugin(t *testing.T, script string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.py"), []byte(script), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

func waitEvent(t *testing.T, events chan plugin_entities.PluginUniversalEvent) plugin_entities.PluginUniversalEvent {
	t.Helper()
	select {
	case event := <-events:
		return event
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for event")
		return plugin_entities.PluginUniversalEvent{}
	}
}

func waitState(t *testing.T, instance *Instance, within time.Duration, accept ...InstanceState) InstanceState {
	t.Helper()
	deadline := time.Now().Add(within)
	for {
		state := instance.State()
		for _, want := range accept {
			if state == want {
				return state
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("state %q never reached %v within %s", state, accept, within)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func waitProcessGone(t *testing.T, workingDir string, within time.Duration) {
	t.Helper()
	pidPath := filepath.Join(workingDir, "pid")
	deadline := time.Now().Add(within)
	var pid int
	for {
		raw, err := os.ReadFile(pidPath)
		if err == nil {
			pid, err = strconv.Atoi(strings.TrimSpace(string(raw)))
			if err == nil && pid > 0 {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("pid file never appeared at %s", pidPath)
		}
		time.Sleep(20 * time.Millisecond)
	}
	for {
		if err := syscall.Kill(pid, 0); err != nil {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("process %d still alive after %s", pid, within)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestInstanceEchoRoundTrip(t *testing.T) {
	instance, err := StartInstance(context.Background(), InstanceConfig{
		WorkingDir:       fixturePluginDir(t),
		PythonPath:       "python3",
		Entrypoint:       "main",
		HeartbeatTimeout: 10 * time.Second,
		Logs:             io.Discard,
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	defer instance.Stop()

	events := make(chan plugin_entities.PluginUniversalEvent, 4)
	instance.Listen("s1", func(event plugin_entities.PluginUniversalEvent) { events <- event })
	if err := instance.Write("s1", InStreamEventRequest, map[string]any{"ping": true}); err != nil {
		t.Fatal(err)
	}

	first := waitEvent(t, events)
	message, _ := first.SessionMessage()
	if message.Type != plugin_entities.SessionMessageStream {
		t.Fatalf("expected stream, got %+v", message)
	}
	var echoed struct {
		Echo  map[string]any `json:"echo"`
		Event string         `json:"event"`
	}
	if err := json.Unmarshal(message.Data, &echoed); err != nil {
		t.Fatalf("unmarshal stream data: %v", err)
	}
	if echoed.Event != string(InStreamEventRequest) {
		t.Fatalf("expected echoed event %q, got %q", InStreamEventRequest, echoed.Event)
	}
	if echoed.Echo["ping"] != true {
		t.Fatalf("expected echoed payload, got %+v", echoed.Echo)
	}
	second := waitEvent(t, events)
	if message, _ = second.SessionMessage(); message.Type != plugin_entities.SessionMessageEnd {
		t.Fatalf("expected end, got %+v", message)
	}
}

func TestInstanceStopKillsProcess(t *testing.T) {
	dir := writePlugin(t, heartbeatingPlugin)
	instance, err := StartInstance(context.Background(), InstanceConfig{
		WorkingDir:       dir,
		PythonPath:       "python3",
		Entrypoint:       "main",
		HeartbeatTimeout: 10 * time.Second,
		Logs:             io.Discard,
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}

	instance.Stop()

	waitState(t, instance, 3*time.Second, InstanceStateStopped)
	waitProcessGone(t, dir, 3*time.Second)
	select {
	case <-instance.Done():
	default:
		t.Fatal("Done channel not closed after Stop")
	}
	if err := instance.Write("s1", InStreamEventRequest, map[string]any{"ping": true}); !errors.Is(err, ErrInstanceStopped) {
		t.Fatalf("expected ErrInstanceStopped, got %v", err)
	}
}

func TestStartInstanceHeartbeatTimeout(t *testing.T) {
	dir := writePlugin(t, silentPlugin)
	started := time.Now()
	_, err := StartInstance(context.Background(), InstanceConfig{
		WorkingDir:       dir,
		PythonPath:       "python3",
		Entrypoint:       "main",
		HeartbeatTimeout: 1 * time.Second,
		Logs:             io.Discard,
	})
	if !errors.Is(err, ErrInstanceStart) {
		t.Fatalf("expected ErrInstanceStart, got %v", err)
	}
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("start took %s, expected ~1s timeout", elapsed)
	}
	waitProcessGone(t, dir, 3*time.Second)
}

func TestWriteAfterStopFails(t *testing.T) {
	instance, err := StartInstance(context.Background(), InstanceConfig{
		WorkingDir:       fixturePluginDir(t),
		PythonPath:       "python3",
		Entrypoint:       "main",
		HeartbeatTimeout: 10 * time.Second,
		Logs:             io.Discard,
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}

	instance.Stop()

	if err := instance.Write("s1", InStreamEventRequest, "payload"); !errors.Is(err, ErrInstanceStopped) {
		t.Fatalf("expected ErrInstanceStopped, got %v", err)
	}
}

func TestUnresponsiveAfterHeartbeatStarvation(t *testing.T) {
	dir := writePlugin(t, singleHeartbeatPlugin)
	instance, err := StartInstance(context.Background(), InstanceConfig{
		WorkingDir:       dir,
		PythonPath:       "python3",
		Entrypoint:       "main",
		HeartbeatTimeout: 1500 * time.Millisecond,
		Logs:             io.Discard,
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	defer instance.Stop()

	if state := instance.State(); state != InstanceStateReady {
		t.Fatalf("expected ready after start, got %q", state)
	}
	waitState(t, instance, 5*time.Second, InstanceStateUnresponsive, InstanceStateStopped)
	waitProcessGone(t, dir, 5*time.Second)
}

func TestStopIsIdempotent(t *testing.T) {
	instance, err := StartInstance(context.Background(), InstanceConfig{
		WorkingDir:       fixturePluginDir(t),
		PythonPath:       "python3",
		Entrypoint:       "main",
		HeartbeatTimeout: 10 * time.Second,
		Logs:             io.Discard,
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}

	instance.Stop()
	instance.Stop()

	if state := instance.State(); state != InstanceStateStopped {
		t.Fatalf("expected stopped, got %q", state)
	}
}
