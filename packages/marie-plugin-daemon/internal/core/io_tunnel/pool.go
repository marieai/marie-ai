package io_tunnel

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	plugindaemon "github.com/marieai/marie-ai/packages/marie-plugin-daemon"
	backwards_invocation "github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/io_tunnel/backwards_invocation"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/local_runtime"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/core/plugin_manager"
	"github.com/marieai/marie-ai/packages/marie-plugin-daemon/internal/marie/decoder"
	plugin_entities "github.com/marieai/marie-ai/packages/marie-plugin-daemon/pkg/entities/plugin_entities"
)

var (
	ErrNotInstalled    = errors.New("plugin_not_installed")
	ErrNotRunning      = errors.New("instance_not_running")
	ErrDeployCancelled = errors.New("deploy_cancelled")
)

const StateAbsent local_runtime.InstanceState = "absent"

const (
	FrameStream = "stream"
	FrameEnd    = "end"
	FrameError  = "error"
	FrameLog    = "log"
)

const (
	sessionBufferSize = 64
	frameBufferSize   = 16
	defaultTimeout    = 30 * time.Second
)

type Frame struct {
	Type string          `json:"type"`
	Data json.RawMessage `json:"data,omitempty"`
}

// deployment is the per-key pool entry. instance and err are written before
// done is closed and may only be read after done is closed.
type deployment struct {
	done       chan struct{}
	workingDir string
	instance   *local_runtime.Instance
	err        error
}

type Pool struct {
	manager *plugin_manager.Manager
	store   *backwards_invocation.Storage
	logs    io.Writer

	mu          sync.Mutex
	deployments map[string]*deployment
}

func NewPool(manager *plugin_manager.Manager, store *backwards_invocation.Storage, logs io.Writer) *Pool {
	if logs == nil {
		logs = io.Discard
	}
	return &Pool{
		manager:     manager,
		store:       store,
		logs:        logs,
		deployments: map[string]*deployment{},
	}
}

func (p *Pool) Deploy(ctx context.Context, tenant, packageRef string) (plugin_manager.Install, error) {
	key := instanceKey(tenant, packageRef)
	for {
		install, ok := p.manager.Get(tenant, packageRef)
		if !ok {
			return plugin_manager.Install{}, fmt.Errorf("%w: %s", ErrNotInstalled, packageRef)
		}

		p.mu.Lock()
		current := p.deployments[key]
		if current == nil {
			mine := &deployment{done: make(chan struct{}), workingDir: install.WorkingDir}
			p.deployments[key] = mine
			p.mu.Unlock()
			return p.runDeploy(ctx, key, tenant, packageRef, install, mine)
		}
		p.mu.Unlock()

		select {
		case <-current.done:
		case <-ctx.Done():
			return plugin_manager.Install{}, ctx.Err()
		}

		if current.err == nil && current.instance != nil && current.workingDir == install.WorkingDir {
			if state := current.instance.State(); state == local_runtime.InstanceStateReady || state == local_runtime.InstanceStateStarting {
				install.State = string(state)
				return install, nil
			}
		}

		// Stale, failed, or superseded by a re-install: retire it and retry.
		p.mu.Lock()
		if p.deployments[key] == current {
			delete(p.deployments, key)
		}
		p.mu.Unlock()
		if current.instance != nil {
			current.instance.Stop()
		}
	}
}

// runDeploy performs the slow work (decode, venv, process start) outside the
// pool mutex; mine is already published in the map as the in-flight marker.
func (p *Pool) runDeploy(ctx context.Context, key, tenant, packageRef string, install plugin_manager.Install, mine *deployment) (plugin_manager.Install, error) {
	fail := func(err error) (plugin_manager.Install, error) {
		mine.err = err
		p.mu.Lock()
		if p.deployments[key] == mine {
			delete(p.deployments, key)
		}
		p.mu.Unlock()
		close(mine.done)
		_ = p.manager.SetState(tenant, packageRef, plugin_manager.StateFailed)
		return plugin_manager.Install{}, err
	}

	if err := p.manager.SetState(tenant, packageRef, plugin_manager.StateStarting); err != nil {
		return fail(err)
	}
	instance, err := p.start(ctx, install)
	if err != nil {
		return fail(err)
	}
	mine.instance = instance

	p.mu.Lock()
	owned := p.deployments[key] == mine
	p.mu.Unlock()
	if !owned {
		mine.err = ErrDeployCancelled
		close(mine.done)
		instance.Stop()
		return plugin_manager.Install{}, ErrDeployCancelled
	}

	if err := p.manager.SetState(tenant, packageRef, plugin_manager.StateReady); err != nil {
		mine.err = err
		p.mu.Lock()
		if p.deployments[key] == mine {
			delete(p.deployments, key)
		}
		p.mu.Unlock()
		close(mine.done)
		instance.Stop()
		return plugin_manager.Install{}, err
	}
	close(mine.done)
	install.State = plugin_manager.StateReady
	return install, nil
}

func (p *Pool) start(ctx context.Context, install plugin_manager.Install) (*local_runtime.Instance, error) {
	decoded, err := decoder.DecodePath(install.WorkingDir)
	if err != nil {
		return nil, err
	}
	runtimeBlock, _ := decoded.Manifest["runtime"].(map[string]any)
	version, _ := runtimeBlock["version"].(string)
	entrypoint, _ := runtimeBlock["entrypoint"].(string)
	if entrypoint == "" {
		entrypoint = "main"
	}

	pythonPath, err := local_runtime.EnsureEnvironment(ctx, install.WorkingDir, version, p.logs)
	if err != nil {
		return nil, err
	}
	pythonRuntimePath, err := plugindaemon.PreparePythonRuntime(install.WorkingDir)
	if err != nil {
		return nil, err
	}

	// The instance must outlive the deploy request, so detach from its context.
	return local_runtime.StartInstance(context.WithoutCancel(ctx), local_runtime.InstanceConfig{
		WorkingDir:        install.WorkingDir,
		PythonPath:        pythonPath,
		PythonRuntimePath: pythonRuntimePath,
		Entrypoint:        entrypoint,
		Logs:              p.logs,
	})
}

// lookup returns the entry for key without ever blocking on in-flight deploys.
func (p *Pool) lookup(key string) (entry *deployment, inflight bool) {
	p.mu.Lock()
	entry = p.deployments[key]
	p.mu.Unlock()
	if entry == nil {
		return nil, false
	}
	select {
	case <-entry.done:
		return entry, false
	default:
		return entry, true
	}
}

func (p *Pool) InstanceState(tenant, packageRef string) local_runtime.InstanceState {
	entry, inflight := p.lookup(instanceKey(tenant, packageRef))
	if entry == nil {
		return StateAbsent
	}
	if inflight {
		return local_runtime.InstanceStateStarting
	}
	if entry.instance == nil {
		return StateAbsent
	}
	return entry.instance.State()
}

func (p *Pool) ReadyCount() int {
	p.mu.Lock()
	entries := make([]*deployment, 0, len(p.deployments))
	for _, entry := range p.deployments {
		entries = append(entries, entry)
	}
	p.mu.Unlock()

	count := 0
	for _, entry := range entries {
		select {
		case <-entry.done:
			if entry.instance != nil && entry.instance.State() == local_runtime.InstanceStateReady {
				count++
			}
		default:
		}
	}
	return count
}

func (p *Pool) Invoke(ctx context.Context, tenant, packageRef string, payload any, timeout time.Duration) (<-chan Frame, error) {
	entry, inflight := p.lookup(instanceKey(tenant, packageRef))
	if entry == nil || inflight || entry.instance == nil || entry.instance.State() != local_runtime.InstanceStateReady {
		return nil, fmt.Errorf("%w: %s", ErrNotRunning, packageRef)
	}
	instance := entry.instance

	sessionID, err := newSessionID()
	if err != nil {
		return nil, err
	}

	events := make(chan plugin_entities.PluginUniversalEvent, sessionBufferSize)
	overflow := make(chan struct{})
	overflowOnce := sync.Once{}
	// The handler runs on the instance reader goroutine and must never block.
	instance.Listen(sessionID, func(event plugin_entities.PluginUniversalEvent) {
		select {
		case events <- event:
		default:
			overflowOnce.Do(func() { close(overflow) })
		}
	})

	if err := instance.Write(sessionID, local_runtime.InStreamEventRequest, payload); err != nil {
		instance.CloseSession(sessionID)
		return nil, err
	}

	if timeout <= 0 {
		timeout = defaultTimeout
	}
	frames := make(chan Frame, frameBufferSize)
	go p.consume(ctx, instance, sessionID, tenant, packageRef, events, overflow, frames, timeout)
	return frames, nil
}

func (p *Pool) consume(
	ctx context.Context,
	instance *local_runtime.Instance,
	sessionID, tenant, packageRef string,
	events <-chan plugin_entities.PluginUniversalEvent,
	overflow <-chan struct{},
	frames chan<- Frame,
	timeout time.Duration,
) {
	defer close(frames)
	defer instance.CloseSession(sessionID)

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	tryEmit := func(frame Frame) {
		select {
		case frames <- frame:
		default:
		}
	}
	errorFrame := func(message string, retryable bool) Frame {
		data, _ := json.Marshal(map[string]any{"message": message, "retryable": retryable})
		return Frame{Type: FrameError, Data: data}
	}
	// emit delivers a frame, aborting on cancel, timeout, or instance death so
	// a slow consumer can never pin the session past the invocation timeout.
	// The timer fires at most once, so every abort path returns immediately.
	emit := func(frame Frame) bool {
		select {
		case frames <- frame:
			return true
		case <-ctx.Done():
			instance.Stop()
			return false
		case <-timer.C:
			instance.Stop()
			tryEmit(errorFrame("invocation timeout after "+timeout.String(), true))
			return false
		case <-instance.Done():
			tryEmit(errorFrame("instance stopped", true))
			return false
		}
	}

	for {
		select {
		case event := <-events:
			message, err := event.SessionMessage()
			if err != nil {
				emit(errorFrame(err.Error(), false))
				return
			}
			switch message.Type {
			case plugin_entities.SessionMessageStream:
				if !emit(Frame{Type: FrameStream, Data: message.Data}) {
					return
				}
			case plugin_entities.SessionMessageEnd:
				emit(Frame{Type: FrameEnd})
				return
			case plugin_entities.SessionMessageError:
				emit(Frame{Type: FrameError, Data: message.Data})
				return
			case plugin_entities.SessionMessageInvoke:
				if !emit(p.handleBackwardsInvocation(instance, sessionID, tenant, packageRef, message.Data)) {
					return
				}
			default:
				emit(errorFrame("unknown session message type: "+string(message.Type), false))
				return
			}
		case <-overflow:
			emit(errorFrame("session buffer overflow: consumer too slow", false))
			return
		case <-timer.C:
			instance.Stop()
			tryEmit(errorFrame("invocation timeout after "+timeout.String(), true))
			return
		case <-instance.Done():
			tryEmit(errorFrame("instance stopped", true))
			return
		case <-ctx.Done():
			instance.Stop()
			return
		}
	}
}

// backwardsInvokeChunk mirrors the upstream invoke payload: type,
// backwards_request_id, and a per-type request body.
type backwardsInvokeChunk struct {
	Type               string `json:"type"`
	BackwardsRequestID string `json:"backwards_request_id"`
	Request            struct {
		Opt   string `json:"opt"`
		Key   string `json:"key"`
		Value string `json:"value"` // hex-encoded, set only
	} `json:"request"`
}

// handleBackwardsInvocation routes a plugin-initiated invoke. The storage
// family executes against the pool's Storage; everything else keeps the
// not-supported rejection.
func (p *Pool) handleBackwardsInvocation(instance *local_runtime.Instance, sessionID, tenant, packageRef string, data json.RawMessage) Frame {
	chunk := backwardsInvokeChunk{}
	_ = json.Unmarshal(data, &chunk)

	if chunk.Type != "storage" {
		return p.rejectBackwardsInvocation(instance, sessionID, chunk.BackwardsRequestID)
	}

	result, err := p.executeStorageOp(tenant, packageRef, chunk)
	if err != nil {
		p.writeBackwardsEvent(instance, sessionID, backwardsErrorEvent(chunk.BackwardsRequestID, err.Error()))
	} else {
		p.writeBackwardsEvent(instance, sessionID, backwardsResponseEvent(chunk.BackwardsRequestID, result))
	}
	p.writeBackwardsEvent(instance, sessionID, backwardsEndEvent(chunk.BackwardsRequestID))

	keyDigest := sha256.Sum256([]byte(chunk.Request.Key))
	logData, _ := json.Marshal(map[string]string{
		"message":              "storage backwards invocation",
		"backwards_request_id": chunk.BackwardsRequestID,
		"op":                   chunk.Request.Opt,
		"key_sha256":           hex.EncodeToString(keyDigest[:]),
	})
	return Frame{Type: FrameLog, Data: logData}
}

// executeStorageOp mirrors the upstream storage handler's response bodies:
// get -> {"data": <hex>}, set -> {"data": "ok"},
// del -> {"data": "ok", "deleted_num": n}, exist -> {"data": bool, "exist_num": n}.
func (p *Pool) executeStorageOp(tenant, packageRef string, chunk backwardsInvokeChunk) (map[string]any, error) {
	if p.store == nil {
		return nil, errors.New("persistence not found")
	}
	key := chunk.Request.Key
	if key == "" {
		return nil, errors.New("storage key is required")
	}

	switch chunk.Request.Opt {
	case "get":
		value, ok, err := p.store.Get(tenant, packageRef, key)
		if err != nil || !ok {
			return nil, errors.New("load data failed, please check if the key is correct or you have not set it")
		}
		return map[string]any{"data": hex.EncodeToString(value)}, nil
	case "set":
		value, err := hex.DecodeString(chunk.Request.Value)
		if err != nil {
			return nil, fmt.Errorf("decode data failed: %s", err.Error())
		}
		if err := p.store.Set(tenant, packageRef, key, value); err != nil {
			return nil, fmt.Errorf("save data failed: %s", err.Error())
		}
		return map[string]any{"data": "ok"}, nil
	case "del":
		_, existed, err := p.store.Get(tenant, packageRef, key)
		if err != nil {
			return nil, fmt.Errorf("delete data failed: %s", err.Error())
		}
		if err := p.store.Delete(tenant, packageRef, key); err != nil {
			return nil, fmt.Errorf("delete data failed: %s", err.Error())
		}
		deleted := 0
		if existed {
			deleted = 1
		}
		return map[string]any{"data": "ok", "deleted_num": deleted}, nil
	case "exist":
		_, existed, err := p.store.Get(tenant, packageRef, key)
		if err != nil {
			return nil, fmt.Errorf("exist data failed: %s", err.Error())
		}
		existNum := 0
		if existed {
			existNum = 1
		}
		return map[string]any{"data": existed, "exist_num": existNum}, nil
	default:
		return nil, fmt.Errorf("unsupported storage opt: %q", chunk.Request.Opt)
	}
}

// Backwards-response bodies mirror the upstream BackwardsInvocationResponseEvent:
// {"backwards_request_id", "event": response|error|end, "message", "data"}.
func backwardsResponseEvent(requestID string, data any) map[string]any {
	return map[string]any{
		"backwards_request_id": requestID,
		"event":                "response",
		"message":              "struct",
		"data":                 data,
	}
}

func backwardsErrorEvent(requestID, message string) map[string]any {
	return map[string]any{
		"backwards_request_id": requestID,
		"event":                "error",
		"message":              message,
		"data":                 nil,
	}
}

func backwardsEndEvent(requestID string) map[string]any {
	return map[string]any{
		"backwards_request_id": requestID,
		"event":                "end",
		"message":              "",
		"data":                 nil,
	}
}

func (p *Pool) writeBackwardsEvent(instance *local_runtime.Instance, sessionID string, event map[string]any) {
	if err := instance.Write(sessionID, local_runtime.InStreamEventBackwardsResponse, event); err != nil {
		fmt.Fprintf(p.logs, "backwards response write failed: %v\n", err)
	}
}

// rejectBackwardsInvocation answers a non-storage invoke with an error.
func (p *Pool) rejectBackwardsInvocation(instance *local_runtime.Instance, sessionID, backwardsRequestID string) Frame {
	if err := instance.Write(sessionID, local_runtime.InStreamEventBackwardsResponse, map[string]any{
		"backwards_request_id": backwardsRequestID,
		"error":                "backwards invocation not supported",
	}); err != nil {
		fmt.Fprintf(p.logs, "backwards response write failed: %v\n", err)
	}
	logData, _ := json.Marshal(map[string]string{
		"message":              "rejected backwards invocation",
		"backwards_request_id": backwardsRequestID,
	})
	return Frame{Type: FrameLog, Data: logData}
}

func (p *Pool) Stop(tenant, packageRef string) error {
	key := instanceKey(tenant, packageRef)
	p.mu.Lock()
	entry := p.deployments[key]
	delete(p.deployments, key)
	p.mu.Unlock()

	if entry == nil {
		return nil
	}
	select {
	case <-entry.done:
	default:
		// In-flight deploy is now disowned; runDeploy stops its own instance.
		return nil
	}
	if entry.instance == nil {
		return nil
	}
	entry.instance.Stop()
	if _, ok := p.manager.Get(tenant, packageRef); ok {
		return p.manager.SetState(tenant, packageRef, plugin_manager.StateStopped)
	}
	return nil
}

func (p *Pool) Shutdown() {
	p.mu.Lock()
	entries := make([]*deployment, 0, len(p.deployments))
	for key, entry := range p.deployments {
		entries = append(entries, entry)
		delete(p.deployments, key)
	}
	p.mu.Unlock()

	for _, entry := range entries {
		<-entry.done
		if entry.instance != nil {
			entry.instance.Stop()
		}
	}
}

func instanceKey(tenant, packageRef string) string {
	return tenant + "/" + packageRef
}

func newSessionID() (string, error) {
	buffer := make([]byte, 16)
	if _, err := rand.Read(buffer); err != nil {
		return "", err
	}
	return hex.EncodeToString(buffer), nil
}
