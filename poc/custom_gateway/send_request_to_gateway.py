import argparse
import asyncio

from docarray import DocList
from docarray.documents import TextDoc

from marie import Client
from marie.serve.runtimes.servers.grpc import GRPCServer


def create_job_submit_request(args: argparse.Namespace):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Job management system")
    parser.add_argument("command", type=str, help="Name of the command to run")
    subparsers = parser.add_subparsers(dest="action", help="Commands", required=True)

    # Status command
    parser_status = subparsers.add_parser("status", help="Check the status of a job")
    parser_status.add_argument(
        "job_id", type=str, help="ID of the job to check status for"
    )

    # Submit command
    parser_submit = subparsers.add_parser("submit", help="Submit a new job")
    parser_submit.add_argument(
        "--no-wait", action="store_true", help="Do not wait for job completion"
    )
    parser_submit.add_argument("script", type=str, help="Script to execute")
    parser_submit.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="Arguments for the script"
    )

    # Logs command
    parser_logs = subparsers.add_parser("logs", help="Get logs for a job")
    parser_logs.add_argument("job_id", type=str, help="ID of the job to get logs for")

    # Stop command
    parser_stop = subparsers.add_parser("stop", help="Stop a running job")
    parser_stop.add_argument("job_id", type=str, help="ID of the job to stop")

    return parser.parse_args()


async def main():
    """
    This function sends a request to a Marie server gateway.
    """
    args = parse_args()

    if args.action == "submit":
        parameters, docs = create_job_submit_request(args)
    elif args.action == "status":
        parameters, docs = create_job_status_request(args)
    elif args.action == "logs":
        parameters, docs = create_job_logs_request(args)
    elif args.action == "stop":
        parameters, docs = create_job_stop_request(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    print(parameters)
    # asyncio.get_event_loop().stop()
    # return

    #  python ./send_request_to_gateway.py status marie_23433
    #  python ./send_request_to_gateway.py submit --no-wait ./script.sh arg1 arg2 arg3
    #  python ./send_request_to_gateway.py logs marie_23433

    client = Client(
        host="127.0.0.1", port=52000, protocol="grpc", request_size=-1, asyncio=True
    )

    ready = await client.is_flow_ready()
    print(f"Flow is ready: {ready}")

    async for resp in client.post(
        on="/",
        inputs=docs,
        parameters=parameters,
        request_size=-1,
        return_responses=True,  # return DocList instead of Response
        return_exceptions=True,
    ):
        print("Response: ")
        print(resp)
        # for doc in resp:G
        #     print(doc.text)
        print(resp.parameters)
        print(resp.data)
        # await asyncio.sleep(1)

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
