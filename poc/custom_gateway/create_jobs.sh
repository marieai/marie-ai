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
    "template_matching": {"enabled": false, "definition_id": "0"}
}
EOF
)

if [ "$#" -lt 2 ]
then
    echo "Please provide the address of the gateway in the format protocol://host:port and the number of jobs to create"
    exit 1
fi

protocol="grpc"
port=""
host=""

# parse the input arguments to get the protocol port and address from URI format  http://127.0.0.1:5100
if [ -z "$1" ]
then
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

if [ -z "$2" ]
then
    echo "Please provide the number of jobs to create"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Using default metadata"
else
    metadata=$(cat "$3")
fi

echo "Using metadata:"
echo "$metadata"

for i in $(seq 1 "$2")
do
    echo "Submitting job $i"
    python ./send_request_to_gateway.py job submit extract --metadata-json "$metadata" --address "$host"  --protocol "$protocol"
    echo "Job $i submitted"
    sleep 1
done