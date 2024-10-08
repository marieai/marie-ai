import argparse
import asyncio

from docarray import DocList
from docarray.documents import TextDoc

from marie import Client


def create_job_submit_request(args: argparse.Namespace):
    param = {
        "invoke_action": {
            "action_type": "command",
            "command": args.command,
            "action": args.action,
            # "job_id": args.job_id,
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def create_job_status_request(args: argparse.Namespace):
    param = {
        "invoke_action": {
            "action_type": "command",
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
            "command": args.command,
            "action": args.action,
            "stream": "stdout",
        }
    }
    docs = DocList[TextDoc]([TextDoc(text=f"Text : {_}") for _ in range(10)])
    return param, docs


def parse_args():
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
    parser_submit = job_subparsers.add_parser("submit", help="Submit a new job")
    parser_submit.add_argument(
        "--no-wait", action="store_true", help="Do not wait for job completion"
    )
    parser_submit.add_argument("script", type=str, help="Script to execute")
    parser_submit.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="Arguments for the script"
    )

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

    return parser.parse_args()


async def main():
    """
    This function sends a request to a Marie server gateway.
    """
    args = parse_args()
    print(args)

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

    client = Client(
        host="127.0.0.1", port=52000, protocol="grpc", request_size=-1, asyncio=True
    )

    ready = await client.is_flow_ready()
    print(f"Flow is ready: {ready}")

    async for resp in client.post(
        on="/",
        inputs=[],  # most request does not need inputs
        parameters=parameters,
        request_size=-1,
        return_responses=True,  # return DocList instead of Response
        return_exceptions=True,
    ):
        print("Response: ")
        print(resp)
        print(resp.parameters)
        print(resp.data)

        ret_docs = resp.data.docs
        for doc in ret_docs:
            print(doc.text)
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
