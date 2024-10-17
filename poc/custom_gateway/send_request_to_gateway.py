import argparse
import asyncio
import os
import time

from docarray import DocList
from docarray.documents import TextDoc

from marie import Client
from marie.utils.json import deserialize_value


def create_job_submit_request(args: argparse.Namespace):
    metadata = deserialize_value(args.metadata_json) if args.metadata_json else {}
    param = {
        "invoke_action": {
            "action_type": "command",
            "api_key": args.api_key,
            "command": args.command,
            "action": args.action,
            "name": args.name,
            "metadata": metadata,
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def create_job_status_request(args: argparse.Namespace):
    param = {
        "invoke_action": {
            "action_type": "command",
            "api_key": args.api_key,
            "command": args.command,
            "action": args.action,
            "job_id": args.job_id,
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def create_job_stop_request(args: argparse.Namespace):
    param = {
        "invoke_action": {
            "action_type": "command",
            "api_key": args.api_key,
            "command": args.command,
            "action": args.action,
            "job_id": args.job_id,
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def create_job_logs_request(args: argparse.Namespace):
    param = {
        "invoke_action": {
            "action_type": "command",
            "api_key": args.api_key,
            "command": args.command,
            "action": args.action,
            "job_id": args.job_id,
            "stream": "stdout",
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def create_job_events_request(args: argparse.Namespace):
    param = {
        "invoke_action": {
            "action_type": "command",
            "api_key": args.api_key,
            "command": args.command,
            "action": args.action,
            "job_id": args.job_id,
            "stream": "stdout",
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def create_job_list_request(args: argparse.Namespace):
    param = {
        "invoke_action": {
            "action_type": "command",
            "api_key": args.api_key,
            "command": args.command,
            "action": args.action,
            "job_id": args.job_id,
            "stream": "stdout",
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def create_nodes_list_request(args: argparse.Namespace):
    param = {
        "invoke_action": {
            "action_type": "command",
            "api_key": args.api_key,
            "command": args.command,
            "action": args.action,
            "stream": "stdout",
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Job management system")
    subparsers = parser.add_subparsers(dest="command", help="Commands", required=True)

    parser_job = subparsers.add_parser("job", help="Manage jobs")
    job_subparsers = parser_job.add_subparsers(
        dest="action", help="Job actions", required=True
    )

    # Status command
    parser_status = job_subparsers.add_parser(
        "status", help="Check the status of a job"
    )
    parser_status.add_argument(
        "job_id", type=str, help="ID of the job to check status for"
    )

    # Submit command
    # ---------------------
    parser_submit = job_subparsers.add_parser("submit", help="Submit a new job")
    parser_submit.add_argument(
        "name", type=str, help="Name of the project this job belongs to"
    )

    parser_submit.add_argument(
        "--address",
        type=str,
        default=os.getenv("MARIE_ADDRESS", "127.0.0.1:52000"),
        help="Address of the Marie cluster to connect to. Can also be specified using the MARIE_ADDRESS environment variable.",
    )

    parser_submit.add_argument(
        "--protocol",
        type=str,
        default=os.getenv("MARIE_PROTOCOL", "grpc"),
        help="Protocol to use for communication with the Marie cluster (grpc, http). Can also be specified using the "
        "MARIE_PROTOCOL environment variable.",
    )

    parser_submit.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("MARIE_API_KEY", None),
        help="API key to use for authentication. Can also be specified using the MARIE_API_KEY environment variable.",
        required=True,
    )

    parser_submit.add_argument(
        "--metadata-json",
        type=str,
        help="JSON-serialized dictionary of metadata to attach to the job.",
    )

    parser_submit.add_argument(
        "--no-wait", action="store_true", help="Do not wait for job completion"
    )

    # parser_submit.add_argument("script", type=str, help="Script to execute")
    # parser_submit.add_argument(
    #     "script_args", nargs=argparse.REMAINDER, help="Arguments for the script"
    # )

    # Logs command
    parser_logs = job_subparsers.add_parser("logs", help="Get logs for a job")
    parser_logs.add_argument("job_id", type=str, help="ID of the job to get logs for")

    # Events command
    parser_logs = job_subparsers.add_parser("events", help="Get events for a job")
    parser_logs.add_argument("job_id", type=str, help="ID of the job to get events for")

    # Stop command
    parser_stop = job_subparsers.add_parser("stop", help="Stop a running job")
    parser_stop.add_argument("job_id", type=str, help="ID of the job to stop")

    parser_job_list = job_subparsers.add_parser("list", help="List all jobs")

    # Node command
    parser_node = subparsers.add_parser("nodes", help="Manage nodes")
    node_subparsers = parser_node.add_subparsers(
        dest="action", help="Node actions", required=True
    )
    # Status command
    nodes_status = node_subparsers.add_parser("list", help="List all nodes")

    return parser.parse_args(args)


async def main():
    """
    This function sends a request to a Marie server gateway.
    """
    # Mock parameters
    mock_args = [
        "job",
        "submit",
        "extract",
        "--address",
        "127.0.0.1:51000",
        "--protocol",
        "http",
        "--api_key",
        "mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ",
    ]
    mock_args = None
    args = parse_args(mock_args)
    print(args)

    parameters = None
    if args.command == "job":
        if args.action == "submit":
            parameters, docs = create_job_submit_request(args)
        elif args.action == "status":
            parameters, docs = create_job_status_request(args)
        elif args.action == "logs":
            parameters, docs = create_job_logs_request(args)
        elif args.action == "events":
            parameters, docs = create_job_events_request(args)
        elif args.action == "stop":
            parameters, docs = create_job_stop_request(args)
        elif args.action == "list":
            parameters, docs = create_job_list_request(args)
    elif args.command == "nodes":
        if args.action == "list":
            parameters, docs = create_nodes_list_request(args)

    if not parameters:
        raise ValueError("Invalid command or action")

    print(parameters)
    # asyncio.get_event_loop().stop()
    # return

    #  python ./send_request_to_gateway.py status marie_23433
    #  python ./send_request_to_gateway.py submit --no-wait ./script.sh arg1 arg2 arg3
    #  python ./send_request_to_gateway.py logs marie_23433
    #  python ./send_request_to_gateway.py job submit hello world

    address = args.address
    protocol = args.protocol
    api_key = args.api_key
    print(f"Connecting to {address} using {protocol} protocol")
    host, port = address.split(":")
    client = Client(
        host=host, port=int(port), protocol=protocol, request_size=-1, asyncio=True
    )

    # ready = await client.is_flow_ready()
    # print(f"Flow is ready: {ready}")
    # auth handler will check the headers, we also need to add the api_key to the metadata
    request_kwargs = {}
    headers = [
        (
            "Authorization",
            f"Bearer {api_key}",
        )
    ]
    request_kwargs["headers"] = headers

    for i in range(0, 1):
        print(f"Sending request : {i}")
        start = time.perf_counter()
        async for resp in client.post(
            on="/api/v1/invoke",
            inputs=[],  # most request does not need inputs
            parameters=parameters,
            request_size=-1,
            return_responses=True,  # return DocList instead of Response
            return_exceptions=True,
            **request_kwargs,  # Unpack request_kwargs here
        ):

            print("Response: ")
            print(resp)
            print(resp.parameters)
            print(resp.data)

            ret_docs = resp.data.docs
            for doc in ret_docs:
                print(doc.text)
            # format
        end = time.perf_counter()
        diff = end - start
        print(f"Time taken: {diff:.6f} seconds")
        print("DONE")

    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(main())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing loop")
        loop.close()

#  ./create_jobs.sh http://127.0.0.1:51000 1 mau_t6qDi1BcL1NkLI8I6iM8z1va0nZP01UQ6LWecpbDz6mbxWgIIIZPfQ
