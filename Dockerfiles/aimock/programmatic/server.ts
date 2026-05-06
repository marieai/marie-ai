/**
 * Marie AI - Programmatic AIMock Server
 *
 * This server provides custom mock implementations for testing Marie AI workflows.
 * Unlike fixture-based mocking, this allows dynamic response generation based on
 * request content, state, and custom logic.
 *
 * Usage:
 *   npm install
 *   npm start
 *
 * Or with Docker:
 *   docker-compose -f docker-compose.mock-llm-programmatic.yml up
 */

import { LLMock } from "@copilotkit/aimock";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";

const PORT = parseInt(process.env.AIMOCK_PORT || "4010", 10);
const ADMIN_PORT = parseInt(process.env.AIMOCK_ADMIN_PORT || "4011", 10);
const VALID_FAULT_PROFILES = new Set(["normal", "timeout", "error", "chaos"]);

type FaultProfile = "normal" | "timeout" | "error" | "chaos";
type MockResponse = Record<string, unknown>;
type MessageHandler = (message: string) => MockResponse | Promise<MockResponse>;

// Document processing state for stateful mocks
const processingState = new Map<string, { status: string; progress: number }>();

const faultState: {
  profile: FaultProfile;
  timeoutMs: number;
  chaosErrorRate: number;
  chaosTimeoutRate: number;
  chaosSlowRate: number;
  chaosSlowMs: number;
} = {
  profile: (process.env.AIMOCK_FAULT_PROFILE as FaultProfile) || "normal",
  timeoutMs: parseInt(process.env.AIMOCK_TIMEOUT_MS || "180000", 10),
  chaosErrorRate: parseFloat(process.env.AIMOCK_CHAOS_ERROR_RATE || "0.15"),
  chaosTimeoutRate: parseFloat(process.env.AIMOCK_CHAOS_TIMEOUT_RATE || "0.15"),
  chaosSlowRate: parseFloat(process.env.AIMOCK_CHAOS_SLOW_RATE || "0.2"),
  chaosSlowMs: parseInt(process.env.AIMOCK_CHAOS_SLOW_MS || "5000", 10),
};

if (!VALID_FAULT_PROFILES.has(faultState.profile)) {
  faultState.profile = "normal";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function applyFaultProfile(): Promise<MockResponse | null> {
  if (faultState.profile === "normal") {
    return null;
  }

  if (faultState.profile === "error") {
    throw new Error("Simulated AIMock error profile");
  }

  if (faultState.profile === "timeout") {
    await sleep(faultState.timeoutMs);
    return {
      content: JSON.stringify({
        status: "delayed",
        profile: "timeout",
        message: "This response was intentionally delayed",
      }),
    };
  }

  const roll = Math.random();
  if (roll < faultState.chaosErrorRate) {
    throw new Error("Simulated AIMock chaos error");
  }
  if (roll < faultState.chaosErrorRate + faultState.chaosTimeoutRate) {
    await sleep(faultState.timeoutMs);
    return {
      content: JSON.stringify({
        status: "delayed",
        profile: "chaos",
        branch: "timeout",
        message: "Chaos timeout branch",
      }),
    };
  }
  if (roll < faultState.chaosErrorRate + faultState.chaosTimeoutRate + faultState.chaosSlowRate) {
    await sleep(faultState.chaosSlowMs);
  }

  return null;
}

function withFaultProfile(handler: MessageHandler | MockResponse): MessageHandler {
  return async (message: string) => {
    const override = await applyFaultProfile();
    if (override) {
      return override;
    }
    return typeof handler === "function" ? await handler(message) : handler;
  };
}

function sendJson(res: ServerResponse, statusCode: number, payload: unknown): void {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload, null, 2));
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  if (chunks.length === 0) {
    return {};
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf-8")) as Record<string, unknown>;
}

function snapshotFaultState(): Record<string, unknown> {
  return {
    profile: faultState.profile,
    timeoutMs: faultState.timeoutMs,
    chaosErrorRate: faultState.chaosErrorRate,
    chaosTimeoutRate: faultState.chaosTimeoutRate,
    chaosSlowRate: faultState.chaosSlowRate,
    chaosSlowMs: faultState.chaosSlowMs,
  };
}

function startAdminServer(): Server {
  const server = createServer(async (req, res) => {
    const method = req.method || "GET";
    const url = new URL(req.url || "/", `http://127.0.0.1:${ADMIN_PORT}`);

    if (method === "GET" && url.pathname === "/health") {
      sendJson(res, 200, { status: "ok", admin: true, ...snapshotFaultState() });
      return;
    }

    if (url.pathname !== "/fault-profile") {
      sendJson(res, 404, { error: "Not found" });
      return;
    }

    if (method === "GET") {
      sendJson(res, 200, snapshotFaultState());
      return;
    }

    if (method !== "POST" && method !== "PUT") {
      sendJson(res, 405, { error: "Method not allowed" });
      return;
    }

    try {
      const body = await readJsonBody(req);
      const requestedProfile = body.profile;
      if (typeof requestedProfile === "string") {
        if (!VALID_FAULT_PROFILES.has(requestedProfile)) {
          sendJson(res, 400, {
            error: "Invalid profile",
            valid_profiles: Array.from(VALID_FAULT_PROFILES),
          });
          return;
        }
        faultState.profile = requestedProfile as FaultProfile;
      }

      if (typeof body.timeoutMs === "number") {
        faultState.timeoutMs = Math.max(1, Math.trunc(body.timeoutMs));
      }
      if (typeof body.chaosErrorRate === "number") {
        faultState.chaosErrorRate = Math.max(0, Math.min(1, body.chaosErrorRate));
      }
      if (typeof body.chaosTimeoutRate === "number") {
        faultState.chaosTimeoutRate = Math.max(0, Math.min(1, body.chaosTimeoutRate));
      }
      if (typeof body.chaosSlowRate === "number") {
        faultState.chaosSlowRate = Math.max(0, Math.min(1, body.chaosSlowRate));
      }
      if (typeof body.chaosSlowMs === "number") {
        faultState.chaosSlowMs = Math.max(1, Math.trunc(body.chaosSlowMs));
      }

      sendJson(res, 200, snapshotFaultState());
    } catch (error) {
      sendJson(res, 400, {
        error: "Invalid JSON body",
        detail: error instanceof Error ? error.message : String(error),
      });
    }
  });

  server.listen(ADMIN_PORT, () => {
    console.log(`AIMock admin server listening on http://localhost:${ADMIN_PORT}`);
  });

  return server;
}

async function main() {
  const mock = new LLMock({ port: PORT });
  const adminServer = startAdminServer();

  // ==========================================================================
  // Document Extraction Handlers
  // ==========================================================================

  // Invoice extraction with dynamic field detection
  mock.onMessage(/extract.*invoice/i, withFaultProfile(async (message: string) => {
    const hasLineItems = message.toLowerCase().includes("line item");
    const hasTotal = message.toLowerCase().includes("total");

    return {
      content: JSON.stringify(
        {
          document_type: "invoice",
          extracted_fields: {
            invoice_number: "INV-2024-" + Math.floor(Math.random() * 10000),
            vendor_name: "Acme Corporation",
            invoice_date: new Date().toISOString().split("T")[0],
            due_date: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000)
              .toISOString()
              .split("T")[0],
            ...(hasTotal && { total_amount: "$1,234.56", currency: "USD" }),
            ...(hasLineItems && {
              line_items: [
                { description: "Service A", quantity: 2, unit_price: 500.0 },
                { description: "Service B", quantity: 1, unit_price: 234.56 },
              ],
            }),
          },
          confidence: 0.95,
          processing_time_ms: Math.floor(Math.random() * 200) + 100,
        },
        null,
        2
      ),
    };
  }));

  // General document extraction
  mock.onMessage(/extract/i, withFaultProfile({
    content: JSON.stringify({
      document_type: "document",
      extracted_fields: {
        title: "Sample Document",
        date: new Date().toISOString().split("T")[0],
        content_summary: "Document content extracted successfully",
      },
      confidence: 0.92,
    }),
  }));

  // ==========================================================================
  // Document Classification Handlers
  // ==========================================================================

  mock.onMessage(/classify/i, withFaultProfile(async (message: string) => {
    // Detect document type hints in the message
    const typeHints: Record<string, { type: string; confidence: number }> = {
      invoice: { type: "invoice", confidence: 0.97 },
      receipt: { type: "receipt", confidence: 0.94 },
      contract: { type: "contract", confidence: 0.91 },
      form: { type: "form", confidence: 0.89 },
      letter: { type: "correspondence", confidence: 0.86 },
    };

    let classification = { type: "unknown", confidence: 0.5 };
    for (const [hint, result] of Object.entries(typeHints)) {
      if (message.toLowerCase().includes(hint)) {
        classification = result;
        break;
      }
    }

    return {
      content: JSON.stringify({
        classification: classification.type,
        confidence: classification.confidence,
        alternative_classifications: [
          { type: "document", confidence: 0.3 },
          { type: "other", confidence: 0.1 },
        ],
      }),
    };
  }));

  // ==========================================================================
  // RAG Query Handlers
  // ==========================================================================

  mock.onMessage(/\b(what|how|why|when|where|who|find|search|query)\b/i, withFaultProfile({
    content:
      "Based on the documents in your knowledge base, here is what I found:\n\n" +
      "The payment terms are Net 30 as specified in the Vendor Agreement (Section 4.2). " +
      "All invoices should be submitted within 5 business days of service completion.\n\n" +
      "**Sources:**\n" +
      "- Vendor Agreement v2.3 (relevance: 95%)\n" +
      "- Accounts Payable Policy (relevance: 88%)",
  }));

  // ==========================================================================
  // Document Summarization Handlers
  // ==========================================================================

  mock.onMessage(/summarize/i, withFaultProfile(async (message: string) => {
    const wordCount = message.split(/\s+/).length;
    const summaryLength = Math.min(Math.floor(wordCount * 0.3), 100);

    return {
      content: JSON.stringify({
        summary:
          "This document outlines the terms and conditions for the business agreement " +
          "between the parties, including payment terms, deliverables, and timelines.",
        key_points: [
          "Agreement effective date: " + new Date().toISOString().split("T")[0],
          "Payment terms: Net 30",
          "Renewal: Annual with 30-day notice",
        ],
        word_count: summaryLength,
        compression_ratio: 0.3,
      }),
    };
  }));

  // ==========================================================================
  // OCR/Text Recognition Handlers
  // ==========================================================================

  mock.onMessage(/ocr|recognize|read.*image|scan/i, withFaultProfile({
    content: JSON.stringify({
      text: "INVOICE\n\nInvoice #: INV-2024-001\nDate: 2024-01-15\n\nBill To:\nAcme Corp\n123 Business St\n\nAmount Due: $1,234.56",
      confidence: 0.94,
      regions: [
        { type: "header", bbox: [0, 0, 100, 20], confidence: 0.98 },
        { type: "table", bbox: [0, 50, 100, 150], confidence: 0.92 },
        { type: "footer", bbox: [0, 180, 100, 200], confidence: 0.89 },
      ],
    }),
  }));

  // ==========================================================================
  // Tool Call Handlers (for function calling)
  // ==========================================================================

  // Document processing tool call
  mock.onMessage(/process.*document|document.*process/i, withFaultProfile({
    content: null,
    tool_calls: [
      {
        id: "call_" + Math.random().toString(36).substr(2, 9),
        type: "function",
        function: {
          name: "process_document",
          arguments: JSON.stringify({
            document_id: "doc_" + Math.random().toString(36).substr(2, 9),
            operations: ["ocr", "extract", "classify"],
            output_format: "json",
          }),
        },
      },
    ],
  }));

  // Search tool call
  mock.onMessage(/search.*knowledge|knowledge.*search/i, withFaultProfile({
    content: null,
    tool_calls: [
      {
        id: "call_" + Math.random().toString(36).substr(2, 9),
        type: "function",
        function: {
          name: "search_knowledge_base",
          arguments: JSON.stringify({
            query: "relevant search query",
            top_k: 5,
            filters: { document_type: "all" },
          }),
        },
      },
    ],
  }));

  // ==========================================================================
  // Streaming Response Handler
  // ==========================================================================

  mock.onMessage(/stream|long.*response/i, withFaultProfile({
    content:
      "This is a streaming response that simulates how real LLM APIs " +
      "return tokens progressively. Each chunk arrives with realistic " +
      "timing based on the configured TPS (tokens per second) and TTFT " +
      "(time to first token) settings.\n\n" +
      "The Marie AI platform uses streaming for:\n" +
      "1. Real-time document processing feedback\n" +
      "2. Progressive extraction results\n" +
      "3. Interactive Q&A sessions\n" +
      "4. Long-form content generation",
    // Streaming is automatic based on aimock.json streaming config
  }));

  // ==========================================================================
  // Default Handler (fallback)
  // ==========================================================================

  mock.onMessage(/.*/i, withFaultProfile({
    content:
      "I'm the Marie AI mock assistant. I can help with:\n" +
      "- Document extraction (try: 'extract invoice data')\n" +
      "- Document classification (try: 'classify this document')\n" +
      "- Knowledge queries (try: 'what are the payment terms?')\n" +
      "- Summarization (try: 'summarize this document')\n" +
      "- OCR (try: 'read text from image')",
  }));

  // ==========================================================================
  // Start Server
  // ==========================================================================

  await mock.start();

  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║  Marie AI Mock Server (Programmatic Mode)                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Server running at: http://localhost:${PORT}                        ║
║                                                                  ║
║  Endpoints:                                                      ║
║    OpenAI:    http://localhost:${PORT}/v1                           ║
║    Anthropic: http://localhost:${PORT}/anthropic                    ║
║    Metrics:   http://localhost:${PORT}/metrics                      ║
║    Admin:     http://localhost:${ADMIN_PORT}/fault-profile              ║
║                                                                  ║
║  Custom Handlers:                                                ║
║    - Document extraction (invoice, general)                      ║
║    - Document classification                                     ║
║    - RAG queries                                                 ║
║    - Summarization                                               ║
║    - OCR/text recognition                                        ║
║    - Tool calls (process_document, search_knowledge_base)        ║
║    - Error simulation                                            ║
║                                                                  ║
║  Press Ctrl+C to stop                                            ║
╚══════════════════════════════════════════════════════════════════╝
`);

  // Handle graceful shutdown
  process.on("SIGINT", async () => {
    console.log("\nShutting down mock server...");
    adminServer.close();
    await mock.stop();
    process.exit(0);
  });

  process.on("SIGTERM", async () => {
    adminServer.close();
    await mock.stop();
    process.exit(0);
  });
}

main().catch((error) => {
  console.error("Failed to start mock server:", error);
  process.exit(1);
});
