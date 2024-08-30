#!/usr/bin/env bash

# create number of jobs based on the input from the user

for i in $(seq 1 $1)
do
    echo "Submitting job $i"
    python ./send_request_to_gateway.py job submit  "hello world - $i"
done