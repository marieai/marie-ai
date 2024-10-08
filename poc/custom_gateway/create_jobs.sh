#!/usr/bin/env bash

# create number of jobs based on the input from the user

metadata=$(cat <<EOF
{
    "on": "/extract",

    "doc_id": "doc_id_$RANDOM",
    "doc_type": "doc_type",
    "uri": "s3://bucket/key",

    "type": "pipeline",
    "name": "default",
    "page_classifier": {"enabled": false},
    "page_splitter": {"enabled": false},
    "page_cleaner": {"enabled": false},
    "page_boundary": {"enabled": false},
    "template_matching": {"enabled": false, "definition_id": "120791"}
}
EOF
)

for i in $(seq 1 "$1")
do
    echo "Submitting job $i"
    python ./send_request_to_gateway.py job submit extract --metadata-json "$metadata" --address 127.0.0.1:52000
    echo "Job $i submitted"
    sleep 1
done