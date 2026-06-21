package runtime

import (
	"encoding/json"
	"fmt"
)

type EventType string

const (
	EventSession   EventType = "session"
	EventHeartbeat EventType = "heartbeat"
	EventError     EventType = "error"
	EventLog       EventType = "log"
)

type SessionMessageType string

const (
	SessionMessageStream SessionMessageType = "stream"
	SessionMessageEnd    SessionMessageType = "end"
	SessionMessageError  SessionMessageType = "error"
	SessionMessageInvoke SessionMessageType = "invoke"
)

type PluginUniversalEvent struct {
	SessionID string          `json:"session_id"`
	Event     EventType       `json:"event"`
	Data      json.RawMessage `json:"data"`
}

type SessionMessage struct {
	Type SessionMessageType `json:"type"`
	Data json.RawMessage    `json:"data"`
}

func ParsePluginUniversalEvent(line []byte) (PluginUniversalEvent, error) {
	var event PluginUniversalEvent
	if err := json.Unmarshal(line, &event); err != nil {
		return event, fmt.Errorf("invalid plugin event: %w", err)
	}
	if event.Event == "" {
		return event, fmt.Errorf("invalid plugin event: missing event type")
	}
	return event, nil
}

func (event PluginUniversalEvent) SessionMessage() (SessionMessage, error) {
	var message SessionMessage
	if err := json.Unmarshal(event.Data, &message); err != nil {
		return message, fmt.Errorf("invalid session message: %w", err)
	}
	return message, nil
}
