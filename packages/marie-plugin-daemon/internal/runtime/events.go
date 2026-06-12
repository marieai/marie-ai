package runtime

import (
	"encoding/json"
	"errors"
	"fmt"
)

var (
	ErrInvalidEvent   = errors.New("invalid_plugin_event")
	ErrInvalidMessage = errors.New("invalid_session_message")
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
		return event, fmt.Errorf("%w: %v", ErrInvalidEvent, err)
	}
	if event.Event == "" {
		return event, fmt.Errorf("%w: missing event type", ErrInvalidEvent)
	}
	return event, nil
}

func (event PluginUniversalEvent) SessionMessage() (SessionMessage, error) {
	var message SessionMessage
	if err := json.Unmarshal(event.Data, &message); err != nil {
		return message, fmt.Errorf("%w: %v", ErrInvalidMessage, err)
	}
	if message.Type == "" {
		return message, fmt.Errorf("%w: missing message type", ErrInvalidMessage)
	}
	return message, nil
}
