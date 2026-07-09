package service

import (
	"encoding/json"
	"net/http"
)

type SSEWriter struct {
	writer  http.ResponseWriter
	flusher http.Flusher
}

func NewSSEWriter(w http.ResponseWriter) *SSEWriter {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	if flusher != nil {
		flusher.Flush()
	}
	return &SSEWriter{writer: w, flusher: flusher}
}

func (s *SSEWriter) WriteData(data any) error {
	body, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := s.writer.Write([]byte("data: ")); err != nil {
		return err
	}
	if _, err := s.writer.Write(body); err != nil {
		return err
	}
	if _, err := s.writer.Write([]byte("\n\n")); err != nil {
		return err
	}
	if s.flusher != nil {
		s.flusher.Flush()
	}
	return nil
}
