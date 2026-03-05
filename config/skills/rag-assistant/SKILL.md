---
name: rag-assistant
description: >
  Answer questions using retrieved context from document collections and knowledge bases.
  Use when the user needs answers grounded in specific documents or data sources.
version: "1.0.0"
license: Apache-2.0
compatibility: Requires marie-ai RAG backend with vector store
allowed-tools: search_documents query_knowledge_base list_collections
user-invokable: true
argument-hint: "[question about documents]"
providers:
  - openai
  - claude
  - vllm
tags:
  - rag
  - knowledge-base
  - qa
  - search
metadata:
  author: marie-ai
  category: knowledge-management
---

# RAG Assistant Skill

Answer questions by retrieving and synthesizing information from document collections and knowledge bases.

## When to Use

Use this skill when:
- User asks questions about their documents or knowledge base
- User wants to find information across multiple documents
- User asks "what do our docs say about..."
- User needs citations or source references
- User wants to query a specific document collection
- User asks questions that require factual grounding

Do NOT use this skill when:
- User is asking general knowledge questions (use general assistant)
- User wants to create or edit documents
- User is working with real-time data (use API tools)
- No relevant document collections exist

## Instructions

1. **Understand the query**:
   - Identify the core question or information need
   - Note any specific document/collection references
   - Determine if user needs summary or specific details

2. **Retrieve relevant context**:
   - Use `search_documents` for semantic search across collections
   - Use `query_knowledge_base` for structured queries
   - Use `list_collections` if user needs to see available sources

3. **Search strategy**:
   - Start with the most specific query
   - If results are insufficient, broaden the search
   - Consider multiple phrasings of the question
   - Check multiple collections if relevant

4. **Synthesize the answer**:
   - Ground your answer in the retrieved context
   - Cite sources with document names and locations
   - Indicate confidence based on context relevance
   - Note if information might be outdated

5. **Handle edge cases**:
   - No relevant results: Inform user, suggest alternatives
   - Conflicting information: Present both views with sources
   - Partial information: Answer what you can, note gaps
   - Out of scope: Explain limitations of the knowledge base

## Examples

**User**: "What's our refund policy?"
**Action**: Search for "refund policy" in policy documents, provide answer with citation

**User**: "Summarize the Q3 report"
**Action**: Retrieve Q3 report sections, provide executive summary with key metrics

**User**: "What documents do we have about API integration?"
**Action**: Search for API integration docs, list relevant documents with descriptions

**User**: "Compare our pricing across product tiers"
**Action**: Retrieve pricing information, create comparison table with sources

## Output Format

```
## Answer

[Clear, direct answer to the question]

### Details

[Supporting information and context]

### Sources

1. **Document Name** (Collection: X, Page: Y)
   > "Relevant quote from the document..."

2. **Another Document** (Collection: Z)
   > "Supporting quote..."

### Confidence

[High/Medium/Low] - [Brief explanation of why]

### Related Information

- [Links to related documents user might find helpful]
```

## Search Best Practices

**Query Formulation**:
- Use key terms from the user's question
- Include synonyms for important concepts
- Consider both specific and general queries

**Result Evaluation**:
- Check semantic similarity scores
- Verify content is actually relevant (not just keyword match)
- Consider recency of documents
- Cross-reference multiple sources

**Citation Standards**:
- Always cite the document name
- Include page/section if available
- Quote relevant passages
- Note if document is dated

## Handling No Results

When search returns no relevant results:

1. Inform the user clearly
2. Suggest alternative queries
3. List available collections
4. Offer to help with a different approach

```
I couldn't find specific information about [topic] in your documents.

**Suggestions**:
- Try searching for: [alternative terms]
- Check these related collections: [list]
- The information might be in: [suggest where it could be]

Would you like me to search for something related?
```

## Quality Assurance

Before providing an answer, verify:
- [ ] Answer is grounded in retrieved context
- [ ] Sources are properly cited
- [ ] Confidence level is appropriate
- [ ] No hallucination beyond retrieved content
- [ ] User's actual question is addressed
