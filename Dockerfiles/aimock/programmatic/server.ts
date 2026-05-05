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

const PORT = parseInt(process.env.AIMOCK_PORT || "4010", 10);

// Document processing state for stateful mocks
const processingState = new Map<string, { status: string; progress: number }>();

async function main() {
  const mock = new LLMock({ port: PORT });

  // ==========================================================================
  // Document Extraction Handlers
  // ==========================================================================

  // Invoice extraction with dynamic field detection
  mock.onMessage(/extract.*invoice/i, async (message: string) => {
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
  });

  // General document extraction
  mock.onMessage(/extract/i, {
    content: JSON.stringify({
      document_type: "document",
      extracted_fields: {
        title: "Sample Document",
        date: new Date().toISOString().split("T")[0],
        content_summary: "Document content extracted successfully",
      },
      confidence: 0.92,
    }),
  });

  // ==========================================================================
  // Document Classification Handlers
  // ==========================================================================

  mock.onMessage(/classify/i, async (message: string) => {
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
  });

  // ==========================================================================
  // RAG Query Handlers
  // ==========================================================================

  mock.onMessage(/\b(what|how|why|when|where|who|find|search|query)\b/i, {
    content:
      "Based on the documents in your knowledge base, here is what I found:\n\n" +
      "The payment terms are Net 30 as specified in the Vendor Agreement (Section 4.2). " +
      "All invoices should be submitted within 5 business days of service completion.\n\n" +
      "**Sources:**\n" +
      "- Vendor Agreement v2.3 (relevance: 95%)\n" +
      "- Accounts Payable Policy (relevance: 88%)",
  });

  // ==========================================================================
  // Document Summarization Handlers
  // ==========================================================================

  mock.onMessage(/summarize/i, async (message: string) => {
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
  });

  // ==========================================================================
  // OCR/Text Recognition Handlers
  // ==========================================================================

  mock.onMessage(/ocr|recognize|read.*image|scan/i, {
    content: JSON.stringify({
      text: "INVOICE\n\nInvoice #: INV-2024-001\nDate: 2024-01-15\n\nBill To:\nAcme Corp\n123 Business St\n\nAmount Due: $1,234.56",
      confidence: 0.94,
      regions: [
        { type: "header", bbox: [0, 0, 100, 20], confidence: 0.98 },
        { type: "table", bbox: [0, 50, 100, 150], confidence: 0.92 },
        { type: "footer", bbox: [0, 180, 100, 200], confidence: 0.89 },
      ],
    }),
  });

  // ==========================================================================
  // Tool Call Handlers (for function calling)
  // ==========================================================================

  // Document processing tool call
  mock.onMessage(/process.*document|document.*process/i, {
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
  });

  // Search tool call
  mock.onMessage(/search.*knowledge|knowledge.*search/i, {
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
  });

  // ==========================================================================
  // Streaming Response Handler
  // ==========================================================================

  mock.onMessage(/stream|long.*response/i, {
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
  });

  // ==========================================================================
  // Error Simulation Handlers
  // ==========================================================================

  mock.onMessage(/error|fail|crash/i, () => {
    throw new Error("Simulated error for testing error handling");
  });

  mock.onMessage(/timeout/i, async () => {
    // Simulate a slow response
    await new Promise((resolve) => setTimeout(resolve, 30000));
    return { content: "This response was intentionally delayed" };
  });

  // ==========================================================================
  // Default Handler (fallback)
  // ==========================================================================

  mock.onMessage(/.*/i, {
    content:
      "I'm the Marie AI mock assistant. I can help with:\n" +
      "- Document extraction (try: 'extract invoice data')\n" +
      "- Document classification (try: 'classify this document')\n" +
      "- Knowledge queries (try: 'what are the payment terms?')\n" +
      "- Summarization (try: 'summarize this document')\n" +
      "- OCR (try: 'read text from image')",
  });

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
    await mock.stop();
    process.exit(0);
  });

  process.on("SIGTERM", async () => {
    await mock.stop();
    process.exit(0);
  });
}

main().catch((error) => {
  console.error("Failed to start mock server:", error);
  process.exit(1);
});
