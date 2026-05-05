/**
 * Marie AI - Vector Database Handlers
 *
 * Custom vector database mock handlers for testing RAG workflows.
 * Provides dynamic similarity search results based on query content.
 */

import { VectorMock } from "@copilotkit/aimock";

// In-memory vector store for stateful testing
const vectorStore = new Map<
  string,
  {
    id: string;
    values: number[];
    metadata: Record<string, unknown>;
  }[]
>();

// Pre-populate with sample documents
const sampleDocuments = [
  {
    id: "doc_vendor_agreement",
    title: "Vendor Agreement v2.3",
    type: "agreement",
    content:
      "Payment terms are Net 30 from invoice date. Late payments incur 1.5% monthly interest.",
    keywords: ["payment", "terms", "vendor", "agreement", "invoice", "net 30"],
  },
  {
    id: "doc_ap_policy",
    title: "Accounts Payable Policy",
    type: "policy",
    content:
      "All vendor invoices must be approved by department head before payment processing.",
    keywords: ["accounts", "payable", "policy", "approval", "invoice"],
  },
  {
    id: "doc_invoice_guide",
    title: "Invoice Processing Guide",
    type: "guide",
    content:
      "Invoices should be submitted via the vendor portal within 5 business days.",
    keywords: ["invoice", "processing", "guide", "portal", "submit"],
  },
  {
    id: "doc_contract_template",
    title: "Standard Contract Template",
    type: "contract",
    content: "This agreement is effective upon signature by both parties.",
    keywords: ["contract", "template", "agreement", "signature", "effective"],
  },
  {
    id: "doc_compliance",
    title: "Compliance Requirements",
    type: "compliance",
    content: "All documents must be retained for 7 years per regulatory requirements.",
    keywords: ["compliance", "retention", "regulatory", "documents", "requirements"],
  },
];

export async function setupVectorHandlers(vector: VectorMock) {
  // ==========================================================================
  // Query Handler with Semantic Matching
  // ==========================================================================

  vector.onQuery("documents", async (query: { vector?: number[]; topK?: number; filter?: Record<string, unknown> }) => {
    const topK = query.topK || 5;
    const filter = query.filter || {};

    // For testing, we'll use keyword matching to simulate semantic search
    // In a real scenario, the query would contain an embedding vector

    // Generate mock matches based on the filter or return top documents
    let matches = sampleDocuments.map((doc, index) => ({
      id: doc.id,
      score: 0.95 - index * 0.05,
      metadata: {
        title: doc.title,
        type: doc.type,
        content_preview: doc.content.substring(0, 100),
      },
    }));

    // Apply type filter if specified
    if (filter.type) {
      matches = matches.filter(
        (m) => sampleDocuments.find((d) => d.id === m.id)?.type === filter.type
      );
    }

    return {
      matches: matches.slice(0, topK),
      namespace: "documents",
    };
  });

  // ==========================================================================
  // Upsert Handler
  // ==========================================================================

  vector.onUpsert("documents", async (vectors: { id: string; values: number[]; metadata?: Record<string, unknown> }[]) => {
    const collection = vectorStore.get("documents") || [];

    for (const vec of vectors) {
      const existing = collection.findIndex((v) => v.id === vec.id);
      if (existing >= 0) {
        collection[existing] = { ...vec, metadata: vec.metadata || {} };
      } else {
        collection.push({ ...vec, metadata: vec.metadata || {} });
      }
    }

    vectorStore.set("documents", collection);

    return {
      upsertedCount: vectors.length,
    };
  });

  // ==========================================================================
  // Delete Handler
  // ==========================================================================

  vector.onDelete("documents", async (ids: string[]) => {
    const collection = vectorStore.get("documents") || [];
    const filtered = collection.filter((v) => !ids.includes(v.id));
    vectorStore.set("documents", filtered);

    return {
      deletedCount: ids.length,
    };
  });

  // ==========================================================================
  // Describe Index Handler
  // ==========================================================================

  vector.onDescribeIndex("documents", async () => {
    const collection = vectorStore.get("documents") || [];

    return {
      name: "documents",
      dimension: 1536,
      metric: "cosine",
      totalVectorCount: sampleDocuments.length + collection.length,
      namespaces: {
        documents: {
          vectorCount: sampleDocuments.length + collection.length,
        },
      },
    };
  });

  // ==========================================================================
  // Embeddings Cache Collection
  // ==========================================================================

  vector.onQuery("embeddings_cache", async (query: { topK?: number }) => {
    // Cache is typically empty or has temporary embeddings
    return {
      matches: [],
      namespace: "embeddings_cache",
    };
  });

  console.log("Vector handlers registered");
}

// Helper function to simulate semantic similarity
function calculateSimilarity(query: string, document: typeof sampleDocuments[0]): number {
  const queryWords = query.toLowerCase().split(/\s+/);
  const matchingKeywords = document.keywords.filter((kw) =>
    queryWords.some((qw) => kw.includes(qw) || qw.includes(kw))
  );
  return Math.min(0.5 + matchingKeywords.length * 0.1, 0.99);
}
