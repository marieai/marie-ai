/**
 * Marie AI - MCP Tool Handlers
 *
 * Custom MCP (Model Context Protocol) handlers for Marie AI testing.
 * These provide dynamic tool execution results based on input parameters.
 */

import { MCPMock } from "@copilotkit/aimock";

export async function setupMCPHandlers(mcp: MCPMock) {
  // ==========================================================================
  // Document Processing Tools
  // ==========================================================================

  mcp.onToolCall("extract_document", async (args: Record<string, unknown>) => {
    const documentId = (args.document_id as string) || "unknown";
    const format = (args.output_format as string) || "json";
    const fields = (args.fields as string[]) || ["all"];

    // Simulate processing time
    await new Promise((resolve) =>
      setTimeout(resolve, Math.random() * 500 + 200)
    );

    const extractedData: Record<string, unknown> = {
      document_id: documentId,
      extraction_timestamp: new Date().toISOString(),
      format: format,
    };

    // Dynamic field extraction based on requested fields
    if (fields.includes("all") || fields.includes("invoice_number")) {
      extractedData.invoice_number =
        "INV-" + Math.floor(Math.random() * 100000);
    }
    if (fields.includes("all") || fields.includes("date")) {
      extractedData.date = new Date().toISOString().split("T")[0];
    }
    if (fields.includes("all") || fields.includes("amount")) {
      extractedData.amount = "$" + (Math.random() * 10000).toFixed(2);
    }
    if (fields.includes("all") || fields.includes("vendor")) {
      extractedData.vendor = "Acme Corporation";
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(
            {
              status: "success",
              data: extractedData,
              confidence: 0.94,
              processing_time_ms: Math.floor(Math.random() * 500) + 200,
            },
            null,
            2
          ),
        },
      ],
    };
  });

  // ==========================================================================
  // Knowledge Base Search
  // ==========================================================================

  mcp.onToolCall(
    "search_knowledge_base",
    async (args: Record<string, unknown>) => {
      const query = (args.query as string) || "";
      const topK = (args.top_k as number) || 5;
      const filters = (args.filters as Record<string, unknown>) || {};

      // Simulate vector search
      const mockResults = [
        {
          id: "doc_001",
          title: "Vendor Agreement v2.3",
          relevance: 0.95,
          snippet:
            "Payment terms are Net 30 from invoice date. Late payments incur 1.5% monthly interest.",
          metadata: { type: "agreement", date: "2024-01-10" },
        },
        {
          id: "doc_002",
          title: "Accounts Payable Policy",
          relevance: 0.88,
          snippet:
            "All vendor invoices must be approved by department head before payment processing.",
          metadata: { type: "policy", date: "2023-11-15" },
        },
        {
          id: "doc_003",
          title: "Invoice Processing Guide",
          relevance: 0.82,
          snippet:
            "Invoices should be submitted via the vendor portal within 5 business days.",
          metadata: { type: "guide", date: "2024-01-05" },
        },
      ];

      // Filter results based on query relevance (simple keyword matching)
      const filteredResults = mockResults
        .filter((doc) => {
          if (
            filters.document_type &&
            filters.document_type !== "all" &&
            doc.metadata.type !== filters.document_type
          ) {
            return false;
          }
          return true;
        })
        .slice(0, topK);

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                query: query,
                results: filteredResults,
                total_matches: filteredResults.length,
                search_time_ms: Math.floor(Math.random() * 50) + 10,
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  // ==========================================================================
  // Document Classification
  // ==========================================================================

  mcp.onToolCall(
    "classify_document",
    async (args: Record<string, unknown>) => {
      const documentId = (args.document_id as string) || "unknown";
      const content = (args.content as string) || "";

      // Simple keyword-based classification
      const classificationRules = [
        { keywords: ["invoice", "amount due", "bill to"], type: "invoice" },
        { keywords: ["agreement", "contract", "terms"], type: "contract" },
        { keywords: ["receipt", "payment received"], type: "receipt" },
        { keywords: ["policy", "procedure", "guidelines"], type: "policy" },
        { keywords: ["form", "application", "submit"], type: "form" },
      ];

      let classification = { type: "document", confidence: 0.5 };
      const lowerContent = content.toLowerCase();

      for (const rule of classificationRules) {
        const matchCount = rule.keywords.filter((kw) =>
          lowerContent.includes(kw)
        ).length;
        if (matchCount > 0) {
          const confidence = Math.min(0.6 + matchCount * 0.15, 0.98);
          if (confidence > classification.confidence) {
            classification = { type: rule.type, confidence };
          }
        }
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                document_id: documentId,
                classification: classification.type,
                confidence: classification.confidence,
                alternatives: [
                  {
                    type: "document",
                    confidence: 1 - classification.confidence,
                  },
                ],
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  // ==========================================================================
  // Workflow Execution
  // ==========================================================================

  const workflowStates = new Map<
    string,
    { status: string; step: number; total: number }
  >();

  mcp.onToolCall("start_workflow", async (args: Record<string, unknown>) => {
    const workflowId = (args.workflow_id as string) || "wf_default";
    const steps = (args.steps as string[]) || ["extract", "classify", "store"];

    workflowStates.set(workflowId, {
      status: "running",
      step: 0,
      total: steps.length,
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(
            {
              workflow_id: workflowId,
              status: "started",
              steps: steps,
              estimated_time_seconds: steps.length * 2,
            },
            null,
            2
          ),
        },
      ],
    };
  });

  mcp.onToolCall(
    "check_workflow_status",
    async (args: Record<string, unknown>) => {
      const workflowId = (args.workflow_id as string) || "wf_default";
      const state = workflowStates.get(workflowId);

      if (!state) {
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ error: "Workflow not found" }),
            },
          ],
        };
      }

      // Progress the workflow
      if (state.step < state.total) {
        state.step++;
      }
      if (state.step >= state.total) {
        state.status = "completed";
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                workflow_id: workflowId,
                status: state.status,
                progress: {
                  current_step: state.step,
                  total_steps: state.total,
                  percentage: Math.round((state.step / state.total) * 100),
                },
              },
              null,
              2
            ),
          },
        ],
      };
    }
  );

  // ==========================================================================
  // Resource Handlers
  // ==========================================================================

  mcp.onResourceRead("documents://recent", async () => {
    return {
      contents: [
        {
          uri: "documents://recent",
          mimeType: "application/json",
          text: JSON.stringify([
            { id: "doc_001", title: "Invoice INV-2024-001", status: "processed" },
            { id: "doc_002", title: "Contract Amendment", status: "pending" },
            { id: "doc_003", title: "Vendor Agreement", status: "processed" },
          ]),
        },
      ],
    };
  });

  mcp.onResourceRead("config://extraction", async () => {
    return {
      contents: [
        {
          uri: "config://extraction",
          mimeType: "application/json",
          text: JSON.stringify({
            default_format: "json",
            supported_formats: ["json", "xml", "csv"],
            max_file_size_mb: 50,
            supported_types: ["pdf", "png", "jpg", "tiff"],
          }),
        },
      ],
    };
  });

  console.log("MCP handlers registered");
}
