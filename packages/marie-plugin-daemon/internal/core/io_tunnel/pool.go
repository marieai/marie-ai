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
	ErrDigestMismatch  = errors.New("package_digest_mismatch")
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

// deployment is the per-key pool entry. install and instance are written before
// done is closed and may only be read after done is closed.
type deployment struct {
	done     chan struct{}
	install  plugin_manager.Install
	instance *local_runtime.Instance
	previous *deployment // protected by Pool.mu; serves requests during replacement
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
	return p.deploy(ctx, tenant, packageRef, nil)
}

// Install prepares a replacement while the current worker remains available.
func (p *Pool) Install(ctx context.Context, tenant string, archive []byte) (plugin_manager.Install, error) {
	packageRef, _, err := plugin_manager.Inspect(archive)
	if err != nil {
		return plugin_manager.Install{}, err
	}
	return p.deploy(ctx, tenant, packageRef, archive)
}

func (p *Pool) deploy(ctx context.Context, tenant, packageRef string, archive []byte) (plugin_manager.Install, error) {
	key := instanceKey(tenant, packageRef)
	for {
		if err := ctx.Err(); err != nil {
			return plugin_manager.Install{}, err
		}
		p.mu.Lock()
		current := p.deployments[key]
		if current != nil {
			select {
			case <-current.done:
			default:
				p.mu.Unlock()
				select {
				case <-current.done:
					continue
				case <-ctx.Done():
					return plugin_manager.Install{}, ctx.Err()
				}
			}
		}
		mine := &deployment{done: make(chan struct{}), previous: current}
		p.deployments[key] = mine
		p.mu.Unlock()
		return p.runDeploy(ctx, key, tenant, packageRef, archive, mine)
	}
}

// runDeploy performs slow preparation outside the pool mutex. The previous
// worker is retired only after the candidate is ready and durably selected.
func (p *Pool) runDeploy(ctx context.Context, key, tenant, packageRef string, archive []byte, mine *deployment) (plugin_manager.Install, error) {
	var install plugin_manager.Install
	var instance *local_runtime.Instance
	previous := mine.previous
	reused := false
	fail := func(err error) (plugin_manager.Install, error) {
		if instance != nil && !reused {
			instance.Stop()
		}
		p.mu.Lock()
		if p.deployments[key] == mine {
			if install.WorkingDir != "" {
				if cleanupErr := p.manager.Discard(install); cleanupErr != nil {
					fmt.Fprintf(p.logs, "failed candidate cleanup: %v\n", cleanupErr)
				}
			}
			if previous != nil {
				p.deployments[key] = previous
			} else {
				delete(p.deployments, key)
			}
			mine.previous = nil
		}
		close(mine.done)
		p.mu.Unlock()
		return plugin_manager.Install{}, err
	}

	var err error
	if archive != nil {
		install, err = p.manager.Prepare(tenant, archive)
	} else {
		var ok bool
		install, ok = p.manager.Get(tenant, packageRef)
		if !ok {
			err = fmt.Errorf("%w: %s", ErrNotInstalled, packageRef)
		}
	}
	if err != nil {
		return fail(err)
	}
	if previous != nil && previous.instance != nil && previous.install.WorkingDir == install.WorkingDir && previous.instance.State() == local_runtime.InstanceStateReady {
		instance = previous.instance
		reused = true
	} else {
		instance, err = p.start(ctx, install)
		if err != nil {
			return fail(err)
		}
	}

	p.mu.Lock()
	if p.deployments[key] != mine {
		p.mu.Unlock()
		return fail(ErrDeployCancelled)
	}
	if err := ctx.Err(); err != nil {
		p.mu.Unlock()
		return fail(err)
	}
	install.State = plugin_manager.StateReady
	if err := p.manager.Activate(install); err != nil {
		p.mu.Unlock()
		return fail(err)
	}
	mine.install = install
	mine.instance = instance
	if previous != nil && previous.instance != nil && previous.instance != instance {
		previous.instance.Stop()
		if err := p.manager.Discard(previous.install); err != nil {
			fmt.Fprintf(p.logs, "retired version cleanup: %v\n", err)
		}
	}
	mine.previous = nil
	close(mine.done)
	p.mu.Unlock()
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
	defer p.mu.Unlock()
	return activeDeployment(p.deployments[key])
}

// Called with Pool.mu held, since an update can change previous at completion.
func activeDeployment(entry *deployment) (*deployment, bool) {
	if entry == nil {
		return nil, false
	}
	select {
	case <-entry.done:
		return entry, false
	default:
		if entry.previous != nil {
			return entry.previous, false
		}
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
		active, _ := activeDeployment(entry)
		entries = append(entries, active)
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

func (p *Pool) Invoke(ctx context.Context, tenant, packageRef, digest string, payload any, timeout time.Duration) (<-chan Frame, error) {
	if timeout <= 0 {
		timeout = defaultTimeout
	}
	deadline := time.Now().Add(timeout)
	entry, inflight := p.lookup(instanceKey(tenant, packageRef))
	if entry == nil || inflight || entry.instance == nil || entry.instance.State() != local_runtime.InstanceStateReady {
		return nil, fmt.Errorf("%w: %s", ErrNotRunning, packageRef)
	}
	if digest != entry.install.Digest {
		return nil, fmt.Errorf("%w: %s", ErrDigestMismatch, packageRef)
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

	invocationCtx, cancel := context.WithDeadline(ctx, deadline)
	if err := instance.Write(invocationCtx, sessionID, local_runtime.InStreamEventRequest, payload); err != nil {
		cancel()
		instance.CloseSession(sessionID)
		if errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil {
			frames := make(chan Frame, 1)
			frames <- invocationTimeoutFrame()
			close(frames)
			return frames, nil
		}
		return nil, err
	}

	frames := make(chan Frame, frameBufferSize)
	go func() {
		defer cancel()
		p.consume(invocationCtx, instance, sessionID, tenant, packageRef, events, overflow, frames)
	}()
	return frames, nil
}

func (p *Pool) consume(
	ctx context.Context,
	instance *local_runtime.Instance,
	sessionID, tenant, packageRef string,
	events <-chan plugin_entities.PluginUniversalEvent,
	overflow <-chan struct{},
	frames chan<- Frame,
) {
	defer close(frames)
	// The protocol cannot cancel plugin work per session. Detach the listener
	// on timeout/cancel; stopping this shared instance would abort other sessions.
	defer instance.CloseSession(sessionID)

	tryEmit := func(frame Frame) {
		select {
		case frames <- frame:
		default:
		}
	}
	contextDone := func() bool {
		if ctx.Err() == nil {
			return false
		}
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			tryEmit(invocationTimeoutFrame())
		}
		return true
	}
	// Output backpressure and backwards writes share the invocation deadline.
	emit := func(frame Frame) bool {
		if contextDone() {
			return false
		}
		select {
		case frames <- frame:
			return true
		case <-ctx.Done():
			contextDone()
			return false
		case <-instance.Done():
			tryEmit(errorFrame("instance stopped", true))
			return false
		}
	}

	for {
		if contextDone() {
			return
		}
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
				if !emit(p.handleBackwardsInvocation(ctx, instance, sessionID, tenant, packageRef, message.Data)) {
					return
				}
			default:
				emit(errorFrame("unknown session message type: "+string(message.Type), false))
				return
			}
		case <-overflow:
			emit(errorFrame("session buffer overflow: consumer too slow", false))
			return
		case <-instance.Done():
			tryEmit(errorFrame("instance stopped", true))
			return
		case <-ctx.Done():
			contextDone()
			return
		}
	}
}

func errorFrame(message string, retryable bool) Frame {
	data, _ := json.Marshal(map[string]any{"message": message, "retryable": retryable})
	return Frame{Type: FrameError, Data: data}
}

func invocationTimeoutFrame() Frame {
	return errorFrame("invocation timeout; plugin execution may continue", false)
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
func (p *Pool) handleBackwardsInvocation(ctx context.Context, instance *local_runtime.Instance, sessionID, tenant, packageRef string, data json.RawMessage) Frame {
	chunk := backwardsInvokeChunk{}
	_ = json.Unmarshal(data, &chunk)

	if chunk.Type != "storage" {
		return p.rejectBackwardsInvocation(ctx, instance, sessionID, chunk.BackwardsRequestID)
	}

	result, err := p.executeStorageOp(tenant, packageRef, chunk)
	if err != nil {
		p.writeBackwardsEvent(ctx, instance, sessionID, backwardsErrorEvent(chunk.BackwardsRequestID, err.Error()))
	} else {
		p.writeBackwardsEvent(ctx, instance, sessionID, backwardsResponseEvent(chunk.BackwardsRequestID, result))
	}
	p.writeBackwardsEvent(ctx, instance, sessionID, backwardsEndEvent(chunk.BackwardsRequestID))

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

func (p *Pool) writeBackwardsEvent(ctx context.Context, instance *local_runtime.Instance, sessionID string, event map[string]any) {
	if err := instance.Write(ctx, sessionID, local_runtime.InStreamEventBackwardsResponse, event); err != nil {
		fmt.Fprintf(p.logs, "backwards response write failed: %v\n", err)
	}
}

// rejectBackwardsInvocation answers a non-storage invoke with an error.
func (p *Pool) rejectBackwardsInvocation(ctx context.Context, instance *local_runtime.Instance, sessionID, backwardsRequestID string) Frame {
	if err := instance.Write(ctx, sessionID, local_runtime.InStreamEventBackwardsResponse, map[string]any{
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
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stop(tenant, packageRef)
}

// Remove fences any pending replacement before deleting its package files.
func (p *Pool) Remove(tenant, packageRef string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.stop(tenant, packageRef); err != nil {
		return err
	}
	return p.manager.Remove(tenant, packageRef)
}

func (p *Pool) stop(tenant, packageRef string) error {
	key := instanceKey(tenant, packageRef)
	entry, inflight := activeDeployment(p.deployments[key])
	delete(p.deployments, key)
	if entry == nil || inflight {
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
	instances := []*local_runtime.Instance{}
	for key, entry := range p.deployments {
		entries = append(entries, entry)
		active, inflight := activeDeployment(entry)
		if !inflight && active.instance != nil {
			instances = append(instances, active.instance)
		}
		delete(p.deployments, key)
	}
	p.mu.Unlock()

	for _, instance := range instances {
		instance.Stop()
	}
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
