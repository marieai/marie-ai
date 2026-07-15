package local_runtime

import (
	"context"
	"encoding/json"
	"io"
	"os/exec"
	"testing"
	"time"

	plugindaemon "github.com/marieai/marie-ai/packages/marie-plugin-daemon"
	plugin_entities "github.com/marieai/marie-ai/packages/marie-plugin-daemon/pkg/entities/plugin_entities"
)

const sharedRuntimePlugin = `from marie_plugins.runtime import run, session_frame

def dispatch(request):
    session_id = request["session_id"]
    return [
        session_frame(session_id, "stream", {"echo": request["data"]}),
        session_frame(session_id, "end", {}),
    ]

run(dispatch)
`

func TestSharedPythonRuntimeRoundTrip(t *testing.T) {
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 not installed")
	}
	dir := writePlugin(t, sharedRuntimePlugin)
	runtimePath, err := plugindaemon.PreparePythonRuntime(dir)
	if err != nil {
		t.Fatalf("prepare runtime failed: %v", err)
	}

	instance, err := StartInstance(context.Background(), InstanceConfig{
		WorkingDir:        dir,
		PythonPath:        "python3",
		PythonRuntimePath: runtimePath,
		Entrypoint:        "main",
		HeartbeatTimeout:  10 * time.Second,
		Logs:              io.Discard,
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}
	defer instance.Stop()

	events := make(chan plugin_entities.PluginUniversalEvent, 2)
	instance.Listen("shared-runtime", func(event plugin_entities.PluginUniversalEvent) { events <- event })
	if err := instance.Write("shared-runtime", InStreamEventRequest, map[string]any{"value": "ok"}); err != nil {
		t.Fatal(err)
	}

	stream, err := waitEvent(t, events).SessionMessage()
	if err != nil {
		t.Fatal(err)
	}
	if stream.Type != plugin_entities.SessionMessageStream {
		t.Fatalf("expected stream, got %q", stream.Type)
	}
	var payload struct {
		Echo map[string]any `json:"echo"`
	}
	if err := json.Unmarshal(stream.Data, &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Echo["value"] != "ok" {
		t.Fatalf("unexpected stream payload: %+v", payload)
	}

	end, err := waitEvent(t, events).SessionMessage()
	if err != nil {
		t.Fatal(err)
	}
	if end.Type != plugin_entities.SessionMessageEnd {
		t.Fatalf("expected end, got %q", end.Type)
	}
}
