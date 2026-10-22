import type { ErrorResponse, TextResponse } from "@copilotkit/aimock";

export function maxTokensReached(): TextResponse {
  return {
    content: '{"document_type":"document","extracted_fields":',
    finishReason: "length",
    usage: {
      prompt_tokens: 128,
      completion_tokens: 16,
      total_tokens: 144,
    },
  };
}

export function repetitiveLengthOutput(): TextResponse {
  return {
    content: '{"document_type":"document","extracted_fields":"' + "loop ".repeat(20),
    finishReason: "length",
    usage: {
      prompt_tokens: 128,
      completion_tokens: 64,
      total_tokens: 192,
    },
  };
}

export function providerUnavailable(message: string): ErrorResponse {
  return {
    error: {
      message,
      type: "server_error",
      code: "provider_unavailable",
    },
    status: 503,
  };
}

export function providerRejected(message: string): ErrorResponse {
  return {
    error: {
      message,
      type: "invalid_request_error",
      code: "provider_rejected",
    },
    status: 422,
  };
}
