/**
 * Marie Node TypeScript Type Definitions
 * Aligned with Cloudflare Workers naming conventions
 */

declare module 'marie:node/console' {
    export enum LogLevel {
        DEBUG = 0,
        INFO = 1,
        WARN = 2,
        ERROR = 3,
    }
    export function log(level: LogLevel, message: string): void;
}

declare module 'marie:node/http-client' {
    export interface HttpRequest {
        method: string;
        url: string;
        headers: [string, string][];
        body?: Uint8Array;
    }

    export interface HttpResponse {
        status: number;
        headers: [string, string][];
        body: Uint8Array;
    }

    export function fetch(req: HttpRequest): HttpResponse;
}

declare module 'marie:node/secrets' {
    export function get(name: string): string | null;
}

declare module 'marie:node/kv' {
    export function get(key: string): Uint8Array | null;
    export function put(key: string, value: Uint8Array, ttlSeconds?: number): void;
    export function del(key: string): void;
}

declare module 'marie:node/events' {
    export function emit(eventType: string, payload: string): void;
}

/**
 * Core types for Marie node execution
 */

/** Data item flowing through the workflow */
export interface Item {
    /** JSON-encoded data payload */
    json: string;
    /** Optional binary data (images, files, etc.) */
    binary?: Uint8Array | number[] | null;
}

/** Environment bindings (similar to Cloudflare's Env) */
export interface Env {
    /** Node configuration as JSON (parsed for you in the wrapper) */
    vars: string;
}

/** Execution context */
export interface Context {
    workflowId: string;
    executionId: string;
    nodeId: string;
    runIndex: number;
}

/** Success response */
export interface ResponseOk {
    ok: Item[];
}

/** Error response */
export interface ResponseErr {
    err: string;
}

/** Response from node execution */
export type Response = ResponseOk | ResponseErr;

/**
 * User's execute function signature
 *
 * @param input - Array of input items from upstream nodes
 * @param env - Environment variables (parsed from JSON)
 * @param ctx - Execution context with workflow/node identifiers
 * @returns Response with ok (output items) or err (error message)
 *
 * @example
 * ```typescript
 * export function execute(input: Item[], env: Record<string, any>, ctx: Context): Response {
 *     return { ok: [{ json: '{"result": "value"}' }] };
 * }
 * ```
 */
export type ExecuteFunction = (
    input: Item[],
    env: Record<string, unknown>,
    ctx: Context
) => Response;
