#!/usr/bin/env bash

usage() {
    echo "Usage: $0 [protocol://host:port] [number_of_jobs] [api_key] [metadata_file]"
    echo
    echo "Arguments:"
    echo "  protocol://host:port  The address of the gateway in the format protocol://host:port"
    echo "  number_of_jobs        The number of jobs to create"
    echo "  api_key               The API key for authentication"
    echo "  metadata_file         (Optional) Path to a JSON file containing metadata"
    echo
    echo "Options:"
    echo "  -h, --help            Show this help message and exit"
    echo
    echo "Example:"
    echo "  $0 http://127.0.0.1:5100 10 your_api_key metadata.json"
}

# Check if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

# create number of jobs based on the input from the user
metadata=$(cat <<EOF
{
    "on__": "/extract",
    "on___": "extract_executor://document/extract",
    "on": "extract_executor://document/extract-direct",
    
    "project_id": "project_id_000001",
    "doc_idXXXX": "doc_id_$RANDOM",
    "ref_id": "doc_id_0001",
    "ref_type": "doc_type",
    "uri": "s3://bucket/key",
    "policy": "allow_all",

    "type": "pipeline",
    "name": "default",
    "page_classifier": {"enabled": false},
    "page_splitter": {"enabled": false},
    "page_cleaner": {"enabled": false},
    "page_boundary": {"enabled": false},
    "template_matching": {"enabled": false, "definition_id": "0"}
}
EOF
)

if [ "$#" -lt 3 ]; then
    echo "Please provide the address of the gateway in the format protocol://host:port, the number of jobs to create, and the API key"
    exit 1
fi

protocol="grpc"
port=""
host=""

# parse the input arguments to get the protocol port and address from URI format  http://127.0.0.1:5100
if [ -z "$1" ]; then
    echo "Please provide the address of the gateway."
    exit 1
else
    address=$(echo "$1" | cut -d ":" -f 2 | cut -d "/" -f 3)
    protocol=$(echo "$1" | cut -d ":" -f 1)
    port=$(echo "$1" | cut -d ":" -f 3 | cut -d "/" -f 1)
    host=$(echo "$1" | cut -d "/" -f 3)
fi

echo "Using address: $address"
echo "Using protocol: $protocol"
echo "Using port: $port"
echo "Using host: $host"

if [ -z "$2" ]; then
    echo "Please provide the number of jobs to create"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Please provide the API key"
    exit 1
fi

api_key="$3"

if [ -z "$4" ]; then
    echo "Using default metadata"
else
    metadata=$(cat "$4")
fi

echo "Using metadata:"
echo "$metadata"

# extract
# mock_simple
# mock_medium
# mock_complex
# mock_with_subgraphs
# mock_parallel_subgraphs 

# ("mock_simple", "Simple Mock Plan"),
# ("mock_medium", "Medium Mock Plan"),
# ("mock_complex", "Complex Mock Plan"),
# ("mock_with_subgraphs", "Mock Plan with Subgraphs"),
# ("mock_parallel_subgraphs", "Mock Plan with Parallel Subgraphs"),
# ("mock_branch_simple", "Simple Branching Mock Plan (JSONPath)"),
# ("mock_switch_complexity", "SWITCH-based Complexity Routing"),
# ("mock_branch_multi_condition", "Multi-Condition Branching (AND/OR)"),
# ("mock_nested_branches", "Nested Branching (Branch within Branch)"),
# ("mock_branch_python_function", "Python Function Branching"),
# ("mock_branch_jsonpath_advanced", "Advanced JSONPath Expressions"),
# ("mock_branch_all_match", "ALL_MATCH Evaluation Mode"),
# ("mock_branch_regex_matching", "Regex Pattern Matching"),

# ("mock_hitl_approval", "HITL Approval Workflow"),
# ("mock_hitl_correction", "HITL Data Correction Workflow"),
# ("mock_hitl_router", "HITL Confidence Router Workflow"),
# ("mock_hitl_complete_workflow", "Complete HITL Workflow (All Node Types)"),

# RAG Indexing Plans:
# ("mock_rag_indexing_simple", "Simple RAG Indexing Pipeline"),
# ("mock_rag_indexing_with_chunking", "RAG Indexing with Chunking"),
# ("mock_rag_indexing_noop", "Minimal RAG Indexing (test only)"),

# RAG Indexing test job functions
submit_rag_pdf() {
    local metadata='{
        "on": "document_backend_executor://extract",
        "uri": "s3://marie-test-bucket/samples/invoice.pdf",
        "ref_id": "rag_test_pdf_001",
        "ref_type": "submission_document",
        "source_id": "submission:test_001",
        "index_name": "test_index",
        "node_type": "document"
    }'
    python ./send_request_to_gateway.py job submit mock_rag_indexing_simple \
        --metadata-json "$metadata" \
        --address "$host" --protocol "$protocol" --api_key "$api_key"
}

submit_rag_docx() {
    local metadata='{
        "on": "document_backend_executor://extract",
        "uri": "s3://marie-test-bucket/samples/contract.docx",
        "ref_id": "rag_test_docx_001",
        "ref_type": "submission_document",
        "source_id": "submission:test_002",
        "index_name": "test_index",
        "node_type": "document"
    }'
    python ./send_request_to_gateway.py job submit mock_rag_indexing_simple \
        --metadata-json "$metadata" \
        --address "$host" --protocol "$protocol" --api_key "$api_key"
}

submit_rag_xlsx() {
    local metadata='{
        "on": "document_backend_executor://extract",
        "uri": "s3://marie-test-bucket/samples/financials.xlsx",
        "ref_id": "rag_test_xlsx_001",
        "ref_type": "submission_document",
        "source_id": "submission:test_003",
        "index_name": "financial_docs",
        "node_type": "spreadsheet"
    }'
    python ./send_request_to_gateway.py job submit mock_rag_indexing_simple \
        --metadata-json "$metadata" \
        --address "$host" --protocol "$protocol" --api_key "$api_key"
}

submit_rag_image() {
    local metadata='{
        "on": "document_backend_executor://extract",
        "uri": "s3://marie-test-bucket/samples/scanned_receipt.jpg",
        "ref_id": "rag_test_image_001",
        "ref_type": "submission_document",
        "source_id": "submission:test_004",
        "index_name": "receipts",
        "node_type": "image"
    }'
    python ./send_request_to_gateway.py job submit mock_rag_indexing_simple \
        --metadata-json "$metadata" \
        --address "$host" --protocol "$protocol" --api_key "$api_key"
}

submit_rag_with_chunking() {
    local metadata='{
        "on": "document_backend_executor://extract",
        "uri": "s3://marie-test-bucket/samples/long_document.pdf",
        "ref_id": "rag_test_chunking_001",
        "ref_type": "submission_document",
        "source_id": "submission:test_005",
        "index_name": "test_index",
        "chunk_size": 512,
        "chunk_overlap": 50
    }'
    python ./send_request_to_gateway.py job submit mock_rag_indexing_with_chunking \
        --metadata-json "$metadata" \
        --address "$host" --protocol "$protocol" --api_key "$api_key"
}

# Default job submission loop (extract jobs)
for i in $(seq 1 "$2"); do
    request_id="job-$i"
    rendered_metadata="${metadata//\{\{job_index\}\}/$i}"
    rendered_metadata="${rendered_metadata//\{\{request_id\}\}/$request_id}"
    rendered_metadata="${rendered_metadata//\{\{timestamp\}\}/$(date +%s)}"
    rendered_metadata="${rendered_metadata//\{\{random\}\}/$RANDOM}"

    echo "Submitting job $i"
    python ./send_request_to_gateway.py job submit mock_simple \
        --metadata-json "$rendered_metadata" \
        --address "$host" --protocol "$protocol" --api_key "$api_key" &
    echo "Job $i submitted"
    sleep 1
done

wait

echo "All requests have completed!"

# Usage examples for RAG indexing tests (uncomment to use):
# submit_rag_pdf
# submit_rag_docx
# submit_rag_xlsx
# submit_rag_image
# submit_rag_with_chunking
