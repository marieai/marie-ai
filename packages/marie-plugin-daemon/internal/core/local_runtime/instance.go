package local_runtime

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	plugin_entities "github.com/marieai/marie-ai/packages/marie-plugin-daemon/pkg/entities/plugin_entities"
)

var (
	ErrInstanceStart   = errors.New("instance_start_failed")
	ErrInstanceStopped = errors.New("instance_stopped")
)

type InstanceState string

const (
	InstanceStateStarting     InstanceState = "starting"
	InstanceStateReady        InstanceState = "ready"
	InstanceStateUnresponsive InstanceState = "unresponsive"
	InstanceStateStopped      InstanceState = "stopped"
)

type InstanceConfig struct {
	WorkingDir        string
	PythonPath        string
	PythonRuntimePath string
	Entrypoint        string
	HeartbeatTimeout  time.Duration
	Logs              io.Writer
}

type SessionHandler func(event plugin_entities.PluginUniversalEvent)

type InStreamEvent string

const (
	InStreamEventRequest           InStreamEvent = "request"
	InStreamEventBackwardsResponse InStreamEvent = "backwards_response"
)

type inStreamMessage struct {
	SessionID      string         `json:"session_id"`
	ConversationID *string        `json:"conversation_id"`
	MessageID      *string        `json:"message_id"`
	AppID          *string        `json:"app_id"`
	EndpointID     *string        `json:"endpoint_id"`
	Context        map[string]any `json:"context"`
	Event          InStreamEvent  `json:"event"`
	Data           any            `json:"data"`
}

type Instance struct {
	cmd   *exec.Cmd
	stdin io.WriteCloser
	logs  io.Writer

	writeMu     sync.Mutex
	stdinClosed bool

	listenersMu sync.Mutex
	listeners   map[string]SessionHandler

	stateMu sync.Mutex
	state   InstanceState

	heartbeats chan struct{}
	readerDone chan struct{}
	done       chan struct{}
	stopOnce   sync.Once
}

func StartInstance(ctx context.Context, config InstanceConfig) (*Instance, error) {
	if config.HeartbeatTimeout == 0 {
		config.HeartbeatTimeout = 120 * time.Second
	}
	if config.Logs == nil {
		config.Logs = io.Discard
	}
	logs := &lockedWriter{w: config.Logs}

	cmd := exec.CommandContext(ctx, config.PythonPath, "-m", config.Entrypoint)
	cmd.Dir = config.WorkingDir
	cmd.Env = scrubbedEnv(config.PythonRuntimePath)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	cmd.Stderr = logs

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("%w: stdin pipe: %v", ErrInstanceStart, err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("%w: stdout pipe: %v", ErrInstanceStart, err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInstanceStart, err)
	}

	instance := &Instance{
		cmd:        cmd,
		stdin:      stdin,
		logs:       logs,
		listeners:  make(map[string]SessionHandler),
		state:      InstanceStateStarting,
		heartbeats: make(chan struct{}, 1),
		readerDone: make(chan struct{}),
		done:       make(chan struct{}),
	}

	go instance.readLoop(stdout)
	go func() {
		<-instance.readerDone
		cmd.Wait()
		close(instance.done)
	}()

	startTimer := time.NewTimer(config.HeartbeatTimeout)
	defer startTimer.Stop()
	select {
	case <-instance.heartbeats:
		if !instance.transition(InstanceStateStarting, InstanceStateReady) {
			instance.Stop()
			return nil, fmt.Errorf("%w: process exited during startup", ErrInstanceStart)
		}
		go instance.watchdog(config.HeartbeatTimeout)
		return instance, nil
	case <-startTimer.C:
		instance.Stop()
		return nil, fmt.Errorf("%w: no heartbeat within %s", ErrInstanceStart, config.HeartbeatTimeout)
	case <-instance.done:
		instance.Stop()
		return nil, fmt.Errorf("%w: process exited before first heartbeat", ErrInstanceStart)
	case <-ctx.Done():
		instance.Stop()
		return nil, fmt.Errorf("%w: %v", ErrInstanceStart, ctx.Err())
	}
}

func (i *Instance) Listen(sessionID string, handler SessionHandler) {
	i.listenersMu.Lock()
	i.listeners[sessionID] = handler
	i.listenersMu.Unlock()
}

func (i *Instance) CloseSession(sessionID string) {
	i.listenersMu.Lock()
	delete(i.listeners, sessionID)
	i.listenersMu.Unlock()
}

func (i *Instance) Write(sessionID string, event InStreamEvent, payload any) error {
	body, err := json.Marshal(inStreamMessage{
		SessionID: sessionID,
		Context:   map[string]any{},
		Event:     event,
		Data:      payload,
	})
	if err != nil {
		return err
	}
	i.writeMu.Lock()
	defer i.writeMu.Unlock()
	if i.stdinClosed || i.State() == InstanceStateStopped {
		return ErrInstanceStopped
	}
	if _, err := i.stdin.Write(append(body, '\n')); err != nil {
		return fmt.Errorf("%w: %v", ErrInstanceStopped, err)
	}
	return nil
}

func (i *Instance) State() InstanceState {
	i.stateMu.Lock()
	defer i.stateMu.Unlock()
	return i.state
}

func (i *Instance) Done() <-chan struct{} {
	return i.done
}

func (i *Instance) Stop() {
	i.stopOnce.Do(func() {
		i.kill()
		i.writeMu.Lock()
		i.stdinClosed = true
		i.stdin.Close()
		i.writeMu.Unlock()
		<-i.done
	})
}

func (i *Instance) readLoop(stdout io.Reader) {
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 64*1024), 5*1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			// Blank line between events on the plugin's stdout — ignore quietly.
			continue
		}
		event, err := plugin_entities.ParsePluginUniversalEvent(line)
		if err != nil {
			// Non-event output on the plugin's stdout (stray prints / framing).
			// The plugin's real logs arrive as EventLog, so drop this quietly.
			continue
		}
		switch event.Event {
		case plugin_entities.EventHeartbeat:
			select {
			case i.heartbeats <- struct{}{}:
			default:
			}
		case plugin_entities.EventSession:
			i.listenersMu.Lock()
			handler := i.listeners[event.SessionID]
			i.listenersMu.Unlock()
			if handler == nil {
				fmt.Fprintf(i.logs, "dropped session event: no listener for %q\n", event.SessionID)
				continue
			}
			handler(event)
		case plugin_entities.EventError, plugin_entities.EventLog:
			fmt.Fprintf(i.logs, "plugin %s: %s\n", event.Event, event.Data)
		default:
			fmt.Fprintf(i.logs, "unhandled plugin event %q\n", event.Event)
		}
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(i.logs, "plugin stdout scanner: %v\n", err)
	}
	i.kill()
	i.markStopped()
	close(i.readerDone)
}

func (i *Instance) watchdog(timeout time.Duration) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case <-i.heartbeats:
			timer.Reset(timeout)
		case <-timer.C:
			if i.transition(InstanceStateReady, InstanceStateUnresponsive) {
				fmt.Fprintf(i.logs, "no heartbeat within %s, killing plugin\n", timeout)
				i.kill()
			}
			return
		case <-i.done:
			return
		}
	}
}

func (i *Instance) transition(from, to InstanceState) bool {
	i.stateMu.Lock()
	defer i.stateMu.Unlock()
	if i.state != from {
		return false
	}
	i.state = to
	return true
}

func (i *Instance) markStopped() {
	i.stateMu.Lock()
	i.state = InstanceStateStopped
	i.stateMu.Unlock()
}

type lockedWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (l *lockedWriter) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.w.Write(p)
}

func (i *Instance) kill() {
	if i.cmd.Process != nil {
		syscall.Kill(-i.cmd.Process.Pid, syscall.SIGKILL)
	}
}

func scrubbedEnv(pythonRuntimePath string) []string {
	env := []string{
		"PATH=" + os.Getenv("PATH"),
		"HOME=" + os.Getenv("HOME"),
		"INSTALL_METHOD=local",
	}
	if pythonRuntimePath != "" {
		env = append(env, "PYTHONPATH="+pythonRuntimePath)
	}
	for _, key := range []string{"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"} {
		if value := os.Getenv(key); value != "" {
			env = append(env, key+"="+value)
		}
	}
	return env
}
