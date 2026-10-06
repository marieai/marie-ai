import assert from "node:assert/strict";
import test from "node:test";

import {
  maxTokensReached,
  providerRejected,
  providerUnavailable,
  repetitiveLengthOutput,
} from "./fault-response.js";

test("providerUnavailable returns an explicit retryable HTTP error", () => {
  assert.deepEqual(providerUnavailable("simulated outage"), {
    error: {
      message: "simulated outage",
      type: "server_error",
      code: "provider_unavailable",
    },
    status: 503,
  });
});

test("providerRejected returns an explicit terminal HTTP error", () => {
  assert.deepEqual(providerRejected("simulated rejection"), {
    error: {
      message: "simulated rejection",
      type: "invalid_request_error",
      code: "provider_rejected",
    },
    status: 422,
  });
});

test("maxTokensReached returns a truncated successful completion", () => {
  assert.deepEqual(maxTokensReached(), {
    content: '{"document_type":"document","extracted_fields":',
    finishReason: "length",
    usage: {
      prompt_tokens: 128,
      completion_tokens: 16,
      total_tokens: 144,
    },
  });
});

test("repetitiveLengthOutput returns a repeated truncated completion", () => {
  const response = repetitiveLengthOutput();

  assert.equal(response.finishReason, "length");
  assert.match(String(response.content), /loop loop loop loop/);
});
