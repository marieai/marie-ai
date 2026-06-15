package plugin_entities

import (
	"errors"
	"testing"
)

func TestParsePluginUniversalEvent(t *testing.T) {
	line := []byte(`{"session_id":"s1","event":"session","data":{"type":"stream","data":{"chunk":1}}}`)
	event, err := ParsePluginUniversalEvent(line)
	if err != nil {
		t.Fatalf("parse failed: %v", err)
	}
	if event.SessionID != "s1" || event.Event != EventSession {
		t.Fatalf("unexpected event: %+v", event)
	}
	message, err := event.SessionMessage()
	if err != nil || message.Type != SessionMessageStream {
		t.Fatalf("unexpected session message: %+v err=%v", message, err)
	}
}

func TestParseHeartbeat(t *testing.T) {
	event, err := ParsePluginUniversalEvent([]byte(`{"session_id":"","event":"heartbeat","data":null}`))
	if err != nil || event.Event != EventHeartbeat {
		t.Fatalf("heartbeat parse failed: %+v err=%v", event, err)
	}
}

func TestRejectInvalidEvent(t *testing.T) {
	if _, err := ParsePluginUniversalEvent([]byte(`not json`)); !errors.Is(err, ErrInvalidEvent) {
		t.Fatalf("expected ErrInvalidEvent, got %v", err)
	}
	if _, err := ParsePluginUniversalEvent([]byte(`{"session_id":"s1","data":null}`)); !errors.Is(err, ErrInvalidEvent) {
		t.Fatalf("expected ErrInvalidEvent for missing event type, got %v", err)
	}
}

func TestRejectSessionMessageWithoutType(t *testing.T) {
	event, err := ParsePluginUniversalEvent([]byte(`{"session_id":"s1","event":"session","data":{"data":{}}}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := event.SessionMessage(); !errors.Is(err, ErrInvalidMessage) {
		t.Fatalf("expected ErrInvalidMessage, got %v", err)
	}
}
