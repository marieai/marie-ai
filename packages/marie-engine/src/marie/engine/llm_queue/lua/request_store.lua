-- Every accessed key is supplied by the caller in the fabric slot.
local a = cjson.decode(ARGV[1])
local op = a.op
if cleanup_only and op ~= 'producer_close' and op ~= 'producer_expire' and
    op ~= 'discard' and op ~= 'purge' and op ~= 'settle' and op ~= 'prune' then
    return redis.error_reply('cleanup operation required')
end
local types = {'hash','string','set','list','hash','hash','string','string',
               'zset','zset','zset','zset','zset','string','set','hash','set'}
for i, expected in ipairs(types) do
    local actual = redis.call('TYPE', KEYS[i]).ok
    if actual ~= 'none' and actual ~= expected then
        return redis.error_reply('invalid store key type')
    end
end
for i=18,#KEYS do
    local actual = redis.call('TYPE',KEYS[i]).ok
    if actual ~= 'none' and actual ~= 'hash' then return redis.error_reply('invalid endpoint registry key') end
end
local R, alive, members, ready, route, usage, owner, generation,
      delayed, processing, deadlines, retention, producers, limits_key, routes, endpoint, endpoints = unpack(KEYS)
local operations = {route_disable=true,initialize=true,owner_acquire=true,owner_renew=true,owner_release=true,circuit_feedback=true,mark_unknown=true,reject_claim=true,producer_create=true,
    producer_renew=true,producer_close=true,producer_expire=true,route=true,endpoint=true,admit=true,
    metadata=true,result=true,claim=true,payload=true,start=true,defer=true,promote=true,
    finish=true,cancel=true,recover=true,discard=true,purge=true,settle=true,prune=true}
if not operations[op] then return redis.error_reply('invalid operation') end
local function integer(value, low, high)
    return type(value) == 'number' and value == math.floor(value) and value >= low and value <= high
end
local function identifier(value)
    return type(value) == 'string' and #value >= 1 and #value <= 128 and string.match(value,'^[%w_-]+$')
end
if type(a.limits) ~= 'table' or type(a.limits_json) ~= 'string' then return redis.error_reply('invalid limits') end
local stored_limits = redis.call('GET',limits_key)
if op ~= 'initialize' and stored_limits ~= a.limits_json then
    return cjson.encode({disposition='invalid_limits'})
end

a.limits = cjson.decode(stored_limits or a.limits_json)

for _, field in ipairs({'max_active_items','max_payload_bytes','max_inline_payload_bytes','max_records',
    'max_storage_bytes','max_ready_ids','max_execution_items','max_execution_bytes','max_producers',
    'max_routes','max_endpoints','result_allowance','metadata_bytes','delivery_grace_ms','claim_lease_ms',
    'remote_uncertainty_ms','max_deadline_ms','max_lease_ms','max_attempts','cleanup_page_size'}) do
    if not integer(a.limits[field],1,2^40) then return redis.error_reply('invalid limit value') end
end
if not identifier(a.fabric_id) or not identifier(a.id) or not identifier(a.producer_id) or not identifier(a.pool_id) then
    return redis.error_reply('invalid identity')
end
for _, field in ipairs({'identity','endpoint_id','revision','config_revision','logical_batch_id',
                        'logical_task_id','reason','claim_id'}) do
    if a[field] ~= nil and not identifier(a[field]) then return redis.error_reply('invalid identity argument') end
end
if op == 'owner_acquire' and not identifier(a.identity) then return redis.error_reply('missing owner identity') end
if op == 'owner_acquire' or op == 'owner_renew' or op == 'producer_create' or op == 'producer_renew' then
    if not integer(a.lease_ms,1,a.limits.max_lease_ms) then return redis.error_reply('invalid lease') end
end
if op == 'claim' or op == 'payload' or op == 'start' or op == 'defer' or op == 'finish' or op == 'settle' then
    if not identifier(a.claim_id) then return redis.error_reply('missing claim identity') end
end
if op == 'defer' or op == 'finish' or op == 'settle' then
    if not integer(a.execution_seq,0,a.limits.max_attempts) then return redis.error_reply('invalid execution sequence') end
end
if op == 'circuit_feedback' and (not identifier(a.claim_id) or not identifier(a.reason) or
    not integer(a.execution_seq,1,a.limits.max_attempts) or not integer(a.open_ms,1,a.limits.max_deadline_ms) or
    (a.outcome ~= 'success' and a.outcome ~= 'unavailable' and a.outcome ~= 'neutral')) then
    return redis.error_reply('invalid circuit feedback') end
if op == 'reject_claim' and (not identifier(a.claim_id) or not identifier(a.reason)) then return redis.error_reply('invalid rejection arguments') end
if op == 'mark_unknown' and (not identifier(a.claim_id) or not identifier(a.reason) or
    not integer(a.execution_seq,1,a.limits.max_attempts)) then return redis.error_reply('invalid unknown arguments') end
if op == 'defer' and (not integer(a.delay_ms,0,a.limits.max_deadline_ms) or not identifier(a.reason) or
    type(a.remote_settled) ~= 'boolean') then return redis.error_reply('invalid defer arguments') end
if op == 'settle' and a.evidence ~= 'remote_completed' and a.evidence ~= 'remote_cancelled' then return redis.error_reply('invalid settlement evidence') end
if op == 'finish' and (type(a.result) ~= 'string' or type(a.success) ~= 'boolean' or
    type(a.oversized) ~= 'boolean' or #a.result > a.limits.result_allowance) then
    return redis.error_reply('invalid result arguments')
end
if op == 'cancel' and type(a.expire) ~= 'boolean' then return redis.error_reply('invalid cancellation arguments') end
if op == 'endpoint' and (not identifier(a.endpoint_id) or
    not integer(a.execution_limit,1,a.limits.max_execution_items) or
    not integer(a.execution_bytes,1,a.limits.max_execution_bytes) or
    (a.gate ~= 'open' and a.gate ~= 'closed')) then return redis.error_reply('invalid endpoint arguments') end
if op == 'route' and (not identifier(a.endpoint_id) or not identifier(a.revision) or
    not integer(a.execution_limit,1,a.limits.max_execution_items) or
    not integer(a.execution_bytes,1,a.limits.max_execution_bytes) or
    (a.enabled ~= '1' and a.enabled ~= '0') or (a.gate ~= 'open' and a.gate ~= 'closed')) then
    return redis.error_reply('invalid route arguments')
end
if op == 'admit' and (type(a.model) ~= 'string' or #a.model < 1 or #a.model > 128 or not identifier(a.endpoint_id) or not identifier(a.config_revision) or
    not identifier(a.logical_batch_id) or not identifier(a.logical_task_id) or
    not integer(a.item_index,0,2^31-1) or not integer(a.cost,1,1000000) or
    not integer(a.expires_at_ms,1,2^53-1) or type(a.payload) ~= 'string' or
    not integer(a.payload_bytes,1,a.limits.max_inline_payload_bytes) or #a.payload ~= a.payload_bytes or
    type(a.digest) ~= 'string' or #a.digest ~= 64) then return redis.error_reply('invalid admission arguments') end
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local function h(key, field) return redis.call('HGET', key, field) end
local function n(key, field) return tonumber(h(key, field) or '0') end
-- HINCRBY accepts canonical decimal integers, not every number accepted by tonumber.
local function stored_integer(raw, maximum)
    if not raw then return true end
    if raw ~= '0' and not string.match(raw,'^[1-9]%d*$') then return false end
    return #raw <= 16 and tonumber(raw) <= maximum
end
local counter_limits = {
    active_items=a.limits.max_active_items, payload_bytes=a.limits.max_payload_bytes,
    records=a.limits.max_records, storage_bytes=a.limits.max_storage_bytes,
    ready_ids=a.limits.max_ready_ids, reserved_items=a.limits.max_execution_items,
    reserved_bytes=a.limits.max_execution_bytes}
for field, maximum in pairs(counter_limits) do
    if not stored_integer(h(usage,field),maximum) then
        return redis.error_reply('invalid store counter')
    end
end
for field, maximum in pairs({failures=3,probe_successes=3,next_probe=2^53-1}) do
    if not stored_integer(h(endpoint,field),maximum) then return redis.error_reply('invalid circuit metadata') end
end
local request_limits = {feedback_seq=a.limits.max_attempts,
    expires_at_ms=2^53-1, execution_seq=a.limits.max_attempts,
    payload_bytes=a.limits.max_inline_payload_bytes,
    storage_charge=a.limits.metadata_bytes+a.limits.result_allowance,
    reserved=1, active=1, reservation_bytes=a.limits.max_execution_bytes,
    uncertainty_until=2^53-1, claim_until=2^53-1, retain_until=2^53-1,
    owner_generation=2^40, result_allowance=a.limits.result_allowance}
for field, maximum in pairs(request_limits) do
    if not stored_integer(h(R,field),maximum) then
        return redis.error_reply('invalid request metadata')
    end
end
local execution_limits = {execution_limit=a.limits.max_execution_items,
    execution_bytes=a.limits.max_execution_bytes, reserved_items=a.limits.max_execution_items,
    reserved_bytes=a.limits.max_execution_bytes}
for field, maximum in pairs(execution_limits) do
    for _, key in ipairs({route, endpoint}) do
        if not stored_integer(h(key,field),maximum) then
            return redis.error_reply('invalid route or endpoint metadata')
        end
    end
end
local stored_generation = redis.call('GET',generation)
if not stored_integer(stored_generation,2^40) or
    (op == 'owner_acquire' and tonumber(stored_generation or '0') >= 2^40) then
    return redis.error_reply('invalid or exhausted owner generation')
end
-- Configured counters are at most 2^40; prospective sums stay below 2^42.
-- Admission/claim capacity checks bound additions; release checks bound subtraction.
if (op == 'claim' or op == 'prune') and redis.call('LINDEX',ready,0) == a.id and
    n(usage,'ready_ids') < 1 then return redis.error_reply('inconsistent ready accounting') end
if redis.call('EXISTS',R) == 1 then
    local state = h(R,'state')
    local states = {ready=true,claimed=true,executing=true,delayed=true,outcome_unknown=true,
        succeeded=true,failed=true,cancelled=true,expired=true,abandoned=true}
    if not states[state] or (state ~= 'abandoned' and h(R,'version') ~= 'v3') then
        return redis.error_reply('invalid request version or state')
    end
    if n(usage,'records') < 1 or n(usage,'storage_bytes') < n(R,'storage_charge') or
        (h(R,'active') == '1' and (n(usage,'active_items') < 1 or n(usage,'payload_bytes') < n(R,'payload_bytes'))) then
        return redis.error_reply('inconsistent request storage accounting')
    end
    if n(R,'reserved') == 1 and op ~= 'admit' then
        local bytes = n(R,'reservation_bytes')
        if n(usage,'reserved_items') < 1 or n(usage,'reserved_bytes') < bytes or
            n(route,'reserved_items') < 1 or n(route,'reserved_bytes') < bytes or
            n(endpoint,'reserved_items') < 1 or n(endpoint,'reserved_bytes') < bytes then
            return redis.error_reply('inconsistent reservation accounting')
        end
    end
end
local function inc(field, delta) redis.call('HINCRBY', usage, field, delta) end
local function reply(disposition)
    return cjson.encode({disposition=disposition,attempt_id=a.id or '',state=h(R,'state') or '',
        execution_seq=n(R,'execution_seq'),expires_at_ms=n(R,'expires_at_ms'),
        cost=n(R,'cost'),payload_bytes=n(R,'payload_bytes')})
end
local function live() return redis.call('EXISTS', alive) == 1 end
local function terminal()
    local s = h(R,'state')
    return s == 'succeeded' or s == 'failed' or s == 'cancelled' or s == 'expired'
end
local function owner_valid() return a.owner and redis.call('GET',owner) == a.owner end
local function route_open()
    local circuit = h(endpoint,'circuit') or 'closed'
    local probe = h(endpoint,'probe_claim')
    local eligible = (not probe or probe == h(R,'claim_id')) and
        (circuit == 'closed' or now >= n(endpoint,'next_probe'))
    return eligible and h(route,'enabled') == '1' and h(endpoint,'gate') == 'open' and
        h(route,'endpoint_id') == h(R,'endpoint_id') and h(route,'revision') == h(R,'config_revision')
end
local function release()
    if h(endpoint,'probe_claim') == h(R,'claim_id') then redis.call('HDEL',endpoint,'probe_claim') end
    if n(R,'reserved') == 1 then
        local bytes = n(R,'reservation_bytes')
        inc('reserved_items',-1); inc('reserved_bytes',-bytes)
        redis.call('HINCRBY',route,'reserved_items',-1)
        redis.call('HINCRBY',route,'reserved_bytes',-bytes)
        redis.call('HINCRBY',endpoint,'reserved_items',-1)
        redis.call('HINCRBY',endpoint,'reserved_bytes',-bytes)
        redis.call('HSET',R,'reserved',0,'reservation_bytes',0)
    end
    redis.call('ZREM',processing,a.id)
end
local function drop_input()
    if h(R,'active') == '1' then
        inc('active_items',-1)
        inc('payload_bytes',-n(R,'payload_bytes'))
        redis.call('HSET',R,'active',0)
    end
    redis.call('HDEL',R,'payload')
end
local function unlink_indexes()
    redis.call('ZREM',delayed,a.id)
    redis.call('ZREM',deadlines,a.id)
    redis.call('ZREM',retention,a.id)
end
local function erase()
    drop_input()
    unlink_indexes()
    redis.call('SREM',members,a.id)
    inc('storage_bytes',-n(R,'storage_charge'))
    inc('records',-1)
    redis.call('DEL',R)
end
local function abandon()
    drop_input()
    unlink_indexes()
    redis.call('SREM',members,a.id)
    if n(R,'reserved') == 1 and n(R,'execution_seq') > 0 and h(R,'state') ~= 'claimed' then
        local values = redis.call('HMGET',R,'producer_id','pool_id','endpoint_id','claim_id',
            'execution_seq','owner_generation','owner_id','reservation_bytes','uncertainty_until','feedback_seq')
        local charge = n(R,'storage_charge')
        redis.call('DEL',R)
        redis.call('HSET',R,'state','abandoned','producer_id',values[1],'pool_id',values[2],
            'endpoint_id',values[3],'claim_id',values[4],'execution_seq',values[5],
            'owner_generation',values[6],'owner_id',values[7],'reservation_bytes',values[8],
            'uncertainty_until',values[9],'feedback_seq',values[10] or '0','reserved',1,'storage_charge',a.limits.metadata_bytes)
        inc('storage_bytes',a.limits.metadata_bytes-charge)
    else
        release()
        erase()
    end
end
local function finish_terminal(state, result)
    drop_input()
    unlink_indexes()
    local charge = a.limits.metadata_bytes + string.len(result)
    inc('storage_bytes',charge-n(R,'storage_charge'))
    local until_time = math.max(n(R,'expires_at_ms'),now+a.limits.delivery_grace_ms)
    redis.call('HSET',R,'state',state,'result',result,'storage_charge',charge,
        'finished_at',now,'retain_until',until_time)
    redis.call('ZADD',retention,until_time,a.id)
end
local dispatcher_ops = {route_disable=true,route=true,endpoint=true,claim=true,start=true,payload=true,defer=true,
    promote=true,finish=true,recover=true,purge=true,settle=true,prune=true,owner_release=true,circuit_feedback=true,mark_unknown=true,reject_claim=true}
if dispatcher_ops[op] and not owner_valid() then return reply('stale_owner') end
if op == 'initialize' then
    if stored_limits and stored_limits ~= a.limits_json then return reply('invalid_limits') end
    redis.call('SET',limits_key,a.limits_json)
    return reply('initialized')
elseif op == 'owner_acquire' then
    local current = redis.call('GET',owner)
    if current then return cjson.encode({disposition='busy'}) end
    local gen = redis.call('INCR',generation)
    redis.call('SET',owner,a.identity..':'..gen,'PX',a.lease_ms)
    return cjson.encode({disposition='acquired',generation=gen})
elseif op == 'owner_renew' then
    if not owner_valid() then return reply('stale_owner') end
    redis.call('PEXPIRE',owner,a.lease_ms)
    return reply('renewed')
elseif op == 'owner_release' then
    redis.call('DEL',owner)
    return reply('released')
elseif op == 'producer_create' then
    if redis.call('ZCARD',producers) >= a.limits.max_producers then return reply('backpressure') end
    if redis.call('ZSCORE',producers,a.producer_id) then return reply('conflict') end
    redis.call('SET',alive,a.producer_id,'PX',a.lease_ms,'NX')
    redis.call('ZADD',producers,now+a.lease_ms,a.producer_id)
    return reply('created')
elseif op == 'producer_renew' then
    if not live() then return reply('producer_dead') end
    redis.call('PEXPIRE',alive,a.lease_ms)
    redis.call('ZADD',producers,now+a.lease_ms,a.producer_id)
    return reply('renewed')
elseif op == 'producer_close' or op == 'producer_expire' then
    if op == 'producer_close' then
        redis.call('DEL',alive)
        if redis.call('SCARD',members) > 0 then redis.call('ZADD',producers,now,a.producer_id) end
    end
    if live() then return reply('producer_live') end
    if redis.call('SCARD',members) == 0 then redis.call('ZREM',producers,a.producer_id) end
    return reply('producer_dead')
elseif op == 'route_disable' then
    if redis.call('SISMEMBER',routes,a.pool_id) == 0 then return reply('missing') end
    redis.call('HSET',route,'enabled','0')
    return reply('disabled')
elseif op == 'endpoint' then
    if a.target_fingerprint and a.target_fingerprint ~= cjson.null then
        if type(a.target_fingerprint) ~= 'string' or #a.target_fingerprint ~= 64 or string.find(a.target_fingerprint,'[^0-9a-f]') then return redis.error_reply('invalid endpoint target') end
        if type(a.registry_ids) ~= 'table' or #a.registry_ids ~= redis.call('SCARD',endpoints) or #KEYS ~= 17+#a.registry_ids then return reply('registry_changed') end
        local seen = {}
        local prefix = string.sub(endpoint,1,#endpoint-#a.endpoint_id)
        for i,id in ipairs(a.registry_ids) do
            if seen[id] or redis.call('SISMEMBER',endpoints,id) ~= 1 or KEYS[17+i] ~= prefix..id then return reply('registry_changed') end
            seen[id] = true
            local target = h(KEYS[17+i],'target_fingerprint')
            if not target and (n(usage,'records') > 0 or n(usage,'reserved_items') > 0) then return reply('binding_migration_required') end
            if target == a.target_fingerprint and id ~= a.endpoint_id then return reply('target_conflict') end
        end
        local current_target = h(endpoint,'target_fingerprint')
        if current_target and current_target ~= a.target_fingerprint then return reply('binding_conflict') end
    end
    if a.transport_fingerprint and a.transport_fingerprint ~= cjson.null then
        if type(a.transport_fingerprint) ~= 'string' or #a.transport_fingerprint ~= 64 or string.find(a.transport_fingerprint,'[^0-9a-f]') then return redis.error_reply('invalid endpoint binding') end
        local binding = h(endpoint,'transport_fingerprint')
        if binding and binding ~= a.transport_fingerprint then return reply('binding_conflict') end
        if not binding and redis.call('EXISTS',endpoint) == 1 and (n(usage,'records') > 0 or n(usage,'reserved_items') > 0) then return reply('binding_migration_required') end
    end
    if redis.call('SISMEMBER',endpoints,a.endpoint_id) == 0 and redis.call('SCARD',endpoints) >= a.limits.max_endpoints then return reply('backpressure') end
    redis.call('SADD',endpoints,a.endpoint_id)
    if a.transport_fingerprint and a.transport_fingerprint ~= cjson.null then redis.call('HSET',endpoint,'transport_fingerprint',a.transport_fingerprint) end
    if a.target_fingerprint and a.target_fingerprint ~= cjson.null then redis.call('HSET',endpoint,'target_fingerprint',a.target_fingerprint) end
    redis.call('HSET',endpoint,'gate',a.gate,'execution_limit',a.execution_limit,'execution_bytes',a.execution_bytes)
    return reply('configured')
elseif op == 'route' then
    if redis.call('SISMEMBER',endpoints,a.endpoint_id) == 0 and redis.call('SCARD',endpoints) >= a.limits.max_endpoints then return reply('backpressure') end
    if redis.call('SISMEMBER',routes,a.pool_id) == 0 and redis.call('SCARD',routes) >= a.limits.max_routes then return reply('backpressure') end
    if n(route,'reserved_items') > 0 and h(route,'endpoint_id') ~= a.endpoint_id then
        return reply('route_busy')
    end
    redis.call('SADD',routes,a.pool_id)
    redis.call('SADD',endpoints,a.endpoint_id)
    redis.call('HSET',endpoint,'gate',a.gate,'execution_limit',a.execution_limit,'execution_bytes',a.execution_bytes)
    redis.call('HSET',route,'endpoint_id',a.endpoint_id,'revision',a.revision,
        'enabled',a.enabled,'gate',a.gate,'execution_limit',a.execution_limit,
        'execution_bytes',a.execution_bytes)
    return reply('configured')
elseif op == 'admit' then
    if redis.call('EXISTS',R) == 1 then
        if h(R,'digest') ~= a.digest or h(R,'producer_id') ~= a.producer_id then return reply('conflict') end
        if not live() then return reply('producer_dead') end
        return reply('existing')
    end
    if not live() then return reply('producer_dead') end
    if a.expires_at_ms <= now or a.expires_at_ms-now > a.limits.max_deadline_ms then return reply('invalid_deadline') end
    if h(route,'enabled') ~= '1' or h(route,'endpoint_id') ~= a.endpoint_id or h(route,'revision') ~= a.config_revision then
        return reply('route_unavailable')
    end
    local charge = a.limits.metadata_bytes + a.limits.result_allowance
    if n(usage,'active_items') >= a.limits.max_active_items or
        n(usage,'payload_bytes')+a.payload_bytes > a.limits.max_payload_bytes or
        n(usage,'records') >= a.limits.max_records or
        n(usage,'storage_bytes')+charge > a.limits.max_storage_bytes or
        n(usage,'ready_ids') >= a.limits.max_ready_ids then return reply('backpressure') end
    redis.call('HSET',R,'version','v3','fabric_group_id',a.fabric_id,'attempt_id',a.id,'producer_id',a.producer_id,'pool_id',a.pool_id,
        'endpoint_id',a.endpoint_id,'config_revision',a.config_revision,'digest',a.digest,
        'logical_batch_id',a.logical_batch_id,'logical_task_id',a.logical_task_id,'item_index',a.item_index,
        'expires_at_ms',a.expires_at_ms,'payload',a.payload,'payload_bytes',a.payload_bytes,
        'model',a.model,'admitted_at_ms',now,'cost',a.cost,'state','ready','execution_seq',0,'active',1,'reserved',0,
        'storage_charge',charge,'result_allowance',a.limits.result_allowance)
    redis.call('SADD',members,a.id)
    redis.call('RPUSH',ready,a.id)
    redis.call('ZADD',deadlines,a.expires_at_ms,a.id)
    inc('active_items',1); inc('payload_bytes',a.payload_bytes)
    inc('records',1); inc('storage_bytes',charge); inc('ready_ids',1)
    return reply('admitted')
end
local exists = redis.call('EXISTS',R) == 1
if exists and (h(R,'producer_id') ~= a.producer_id or h(R,'pool_id') ~= a.pool_id) then
    return reply('conflict')
end
if op == 'claim' or op == 'prune' then
    if redis.call('LINDEX',ready,0) == a.id and (not exists or terminal() or h(R,'state') == 'abandoned') then
        redis.call('LPOP',ready); inc('ready_ids',-1)
        return reply('tombstone')
    end
end
if op == 'result' and not live() then return reply('producer_dead') end
if not exists then return reply('missing') end
if op == 'metadata' then return reply('found') end
if op == 'circuit_feedback' then
    if h(R,'claim_id') ~= a.claim_id or n(R,'execution_seq') ~= a.execution_seq then return reply('stale_claim') end
    if n(R,'feedback_seq') >= a.execution_seq then return reply('existing') end
    local circuit = h(endpoint,'circuit') or 'closed'
    local probe = h(endpoint,'probe_claim') == a.claim_id
    if a.outcome == 'unavailable' then
        local failures = math.min(n(endpoint,'failures')+1,3)
        redis.call('HSET',endpoint,'failures',failures,'category',a.reason)
        if probe or failures >= 3 then
            redis.call('HSET',endpoint,'circuit','open','next_probe',now+a.open_ms,'probe_successes',0)
        end
    elseif a.outcome == 'success' then
        if probe then
            local successes = n(endpoint,'probe_successes')+1
            redis.call('HSET',endpoint,'probe_successes',successes,'circuit',successes >= 3 and 'closed' or 'half_open')
            if successes >= 3 then redis.call('HSET',endpoint,'failures',0) end
        elseif circuit == 'closed' then redis.call('HSET',endpoint,'failures',0) end
    elseif probe then
        redis.call('HSET',endpoint,'circuit','open','next_probe',now+a.open_ms,'probe_successes',0)
    end
    redis.call('HSET',R,'feedback_seq',a.execution_seq)
    return reply('recorded')
end
if op == 'mark_unknown' then
    if h(R,'claim_id') ~= a.claim_id or n(R,'execution_seq') ~= a.execution_seq then return reply('stale_claim') end
    if h(R,'state') ~= 'executing' and h(R,'state') ~= 'outcome_unknown' then return reply('invalid_state') end
    redis.call('HSET',R,'state','outcome_unknown','last_error',a.reason)
    return reply('outcome_unknown')
end
if op == 'result' then
    if not live() then return reply('producer_dead') end
    if terminal() then return cjson.encode({disposition='result',result=h(R,'result')}) end
    return reply('pending')
end
if op == 'discard' then
    if live() then return reply('producer_live') end
    abandon()
    return reply('discarded')
end
if op == 'settle' then
    if h(R,'claim_id') ~= a.claim_id or n(R,'execution_seq') ~= a.execution_seq then return reply('stale_claim') end
    local state = h(R,'state')
    if (state ~= 'executing' and state ~= 'outcome_unknown' and state ~= 'abandoned' and not terminal()) or
        n(R,'execution_seq') < 1 or n(R,'uncertainty_until') < 1 then return reply('invalid_state') end
    if n(R,'reserved') == 0 then return reply('existing') end
    if n(R,'reservation_bytes') < 1 then return reply('invalid_state') end
    release()
    if h(R,'state') == 'abandoned' then erase() end
    return reply('settled')
end
if op == 'purge' then
    if not terminal() then return reply('invalid_state') end
    if live() and now < n(R,'retain_until') then return reply('not_due') end
    abandon()
    return reply('purged')
end
if not live() then
    if op == 'payload' or op == 'start' or op == 'finish' then return reply('producer_dead') end
    abandon()
    return reply('producer_dead')
end
if op == 'cancel' then
    if terminal() then return reply('existing') end
    if a.expire and now < n(R,'expires_at_ms') then return reply('not_due') end
    if h(R,'state') == 'claimed' then release() end
    local result = {error=a.expire and 'expired' or 'cancelled'}
    if h(R,'last_error') then result.category = h(R,'last_error') end
    finish_terminal(a.expire and 'expired' or 'cancelled',cjson.encode(result))
    return reply('finished')
end
if terminal() then
    if op == 'finish' and h(R,'claim_id') == a.claim_id and n(R,'execution_seq') == a.execution_seq then return reply('existing') end
    return reply('terminal')
end
if now >= n(R,'expires_at_ms') then return reply('expired') end
local state = h(R,'state')
local function claim_matches()
    return h(R,'claim_id') == a.claim_id and h(R,'owner_id') == a.owner
end
if op == 'claim' then
    if state == 'claimed' and claim_matches() then return reply('existing') end
    if state ~= 'ready' then return reply('invalid_state') end
    if redis.call('LINDEX',ready,0) ~= a.id then return reply('head_changed') end
    if a.expected_cost ~= nil and a.expected_cost ~= cjson.null and a.expected_cost ~= n(R,'cost') then return reply('cost_changed') end
    if not route_open() then return reply('gated') end
    local bytes = n(R,'payload_bytes')
    if n(route,'reserved_items') >= n(route,'execution_limit') or
        n(route,'reserved_bytes')+bytes > n(route,'execution_bytes') or
        n(endpoint,'reserved_items') >= n(endpoint,'execution_limit') or
        n(endpoint,'reserved_bytes')+bytes > n(endpoint,'execution_bytes') or
        n(usage,'reserved_items') >= a.limits.max_execution_items or
        n(usage,'reserved_bytes')+bytes > a.limits.max_execution_bytes then return reply('capacity') end
    if (h(endpoint,'circuit') or 'closed') ~= 'closed' then
        redis.call('HSET',endpoint,'circuit','half_open','probe_claim',a.claim_id)
    end
    redis.call('LPOP',ready); inc('ready_ids',-1)
    redis.call('HSET',R,'state','claimed','claim_id',a.claim_id,'owner_id',a.owner,
        'owner_generation',a.generation,'claim_until',now+a.limits.claim_lease_ms,
        'reserved',1,'reservation_bytes',bytes)
    redis.call('ZADD',processing,now+a.limits.claim_lease_ms,a.id)
    inc('reserved_items',1); inc('reserved_bytes',bytes)
    redis.call('HINCRBY',route,'reserved_items',1)
    redis.call('HINCRBY',route,'reserved_bytes',bytes)
    redis.call('HINCRBY',endpoint,'reserved_items',1)
    redis.call('HINCRBY',endpoint,'reserved_bytes',bytes)
    return reply('claimed')
elseif op == 'recover' then
    if state ~= 'claimed' and state ~= 'executing' and state ~= 'outcome_unknown' then return reply('invalid_state') end
    if h(R,'owner_id') == a.owner and now < n(R,'claim_until') then return reply('not_due') end
    if state == 'claimed' then
        if n(usage,'ready_ids') >= a.limits.max_ready_ids then return reply('backpressure') end
        release()
        redis.call('HSET',R,'state','ready')
        redis.call('HDEL',R,'claim_id','owner_id','owner_generation')
        redis.call('RPUSH',ready,a.id); inc('ready_ids',1)
        return reply('requeued')
    end
    redis.call('HSET',R,'state','outcome_unknown','owner_id',a.owner,'owner_generation',a.generation)
    redis.call('ZADD',processing,n(R,'uncertainty_until'),a.id)
    return reply('outcome_unknown')
elseif op == 'promote' then
    if state ~= 'delayed' then return reply('invalid_state') end
    if now < n(R,'next_eligible') then return reply('not_due') end
    if not route_open() then return reply('gated') end
    if n(usage,'ready_ids') >= a.limits.max_ready_ids then return reply('backpressure') end
    redis.call('ZREM',delayed,a.id)
    redis.call('RPUSH',ready,a.id); inc('ready_ids',1)
    redis.call('HSET',R,'state','ready')
    return reply('promoted')
end
if not claim_matches() then return reply('stale_claim') end
if op == 'reject_claim' then
    if state ~= 'claimed' then return reply('invalid_state') end
    release()
    finish_terminal('failed',cjson.encode({error=a.reason}))
    return reply('finished')
end
if op == 'payload' or op == 'start' then
    if state == 'executing' and op == 'start' then return reply('already_executing') end
    if state ~= 'claimed' or now >= n(R,'claim_until') then return reply('stale_claim') end
    if n(R,'reserved') ~= 1 or n(R,'reservation_bytes') < 1 or
        n(R,'reservation_bytes') ~= n(R,'payload_bytes') then return reply('invalid_state') end
    if not route_open() then return reply('gated') end
    if op == 'payload' then return cjson.encode({disposition='payload',payload=h(R,'payload')}) end
    if n(R,'execution_seq') >= a.limits.max_attempts then return reply('attempts_exhausted') end
    redis.call('HSET',R,'state','executing','uncertainty_until',now+a.limits.remote_uncertainty_ms)
    redis.call('HINCRBY',R,'execution_seq',1)
    redis.call('ZADD',processing,now+a.limits.remote_uncertainty_ms,a.id)
    return reply('started')
end
if n(R,'execution_seq') ~= a.execution_seq then return reply('stale_claim') end
if op == 'defer' then
    if state ~= 'claimed' and state ~= 'executing' then return reply('invalid_state') end
    if state == 'executing' and not a.remote_settled then return reply('unresolved') end
    if n(R,'execution_seq') >= a.limits.max_attempts then return reply('attempts_exhausted') end
    release()
    redis.call('HSET',R,'state','delayed','next_eligible',math.min(now+a.delay_ms,n(R,'expires_at_ms')),'last_error',a.reason)
    redis.call('ZADD',delayed,math.min(now+a.delay_ms,n(R,'expires_at_ms')),a.id)
    return reply('deferred')
elseif op == 'finish' then
    if state ~= 'executing' and state ~= 'outcome_unknown' then return reply('invalid_state') end
    release()
    if a.oversized or string.len(a.result) > n(R,'result_allowance') then
        finish_terminal('failed','{"error":"result_too_large"}')
        return reply('result_too_large')
    end
    finish_terminal(a.success and 'succeeded' or 'failed',a.result)
    return reply('finished')
end
return reply('invalid_state')
