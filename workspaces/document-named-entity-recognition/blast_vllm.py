import argparse
import asyncio
import json
import os
import time
import uuid

import httpx


# qwen_v3_30b_instruct
def load_payload(path):
    with open(path, "r") as f:
        data = json.load(f)
    # If it's a list, wrap it in an OpenAI-style payload
    if isinstance(data, list):
        return {"model": "qwen_v3_30b_instruct", "messages": data, "max_tokens": 4096}
    elif isinstance(data, dict) and "messages" in data:
        return data
    else:
        raise ValueError(
            "Invalid format: expected dict with 'messages' or list of message objects."
        )


def maybe_randomize(payload, randomize):
    if not randomize:
        return payload
    p = json.loads(json.dumps(payload))  # deep copy
    # Add a tiny unique user turn to avoid prefix-cache hits (optional)
    tag = f"RAND#{uuid.uuid4().hex[:8]}"
    p["messages"].append({"role": "user", "content": f"random tag: {tag}"})
    return p


async def one_call(client, url, payload):
    t0 = time.perf_counter()
    r = await client.post(url, json=payload, timeout=None)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    js = r.json()
    usage = js.get("usage", {})  # vLLM often returns usage
    prompt = usage.get("prompt_tokens", 0)
    comp = usage.get("completion_tokens", 0)
    return dt, prompt, comp


async def run(url, payload, total, conc, randomize, outdir):
    # Create output directory if not exists
    os.makedirs(outdir, exist_ok=True)
    print(f"📂 Saving individual responses to: {outdir}")

    sem = asyncio.Semaphore(conc)
    prompt_tok = 0
    comp_tok = 0
    latencies = []

    # Configure connection limits to support high concurrency
    limits = httpx.Limits(max_keepalive_connections=conc, max_connections=conc * 2)
    async with httpx.AsyncClient(
        http2=False,
        headers={"Content-Type": "application/json"},
        limits=limits,
        timeout=httpx.Timeout(300.0, connect=60.0),
    ) as client:

        async def worker(i):
            nonlocal prompt_tok, comp_tok
            async with sem:
                pl = maybe_randomize(payload, randomize)
                t0 = time.perf_counter()
                r = await client.post(url, json=pl, timeout=None)
                dt = time.perf_counter() - t0

                # Save JSON response to per-request file
                if r.status_code == 200:
                    js = r.json()
                    fname = os.path.join(
                        outdir, f"resp_{i:05d}_{uuid.uuid4().hex[:8]}.json"
                    )
                    with open(fname, "w") as f:
                        json.dump(js, f, ensure_ascii=False, indent=2)
                    usage = js.get("usage", {})
                    prompt_tok += usage.get("prompt_tokens", 0)
                    comp_tok += usage.get("completion_tokens", 0)
                else:
                    err_path = os.path.join(outdir, f"error_{i:05d}.txt")
                    with open(err_path, "w") as f:
                        f.write(f"Status {r.status_code}\n{r.text}")
                latencies.append(dt)

        t0 = time.perf_counter()
        await asyncio.gather(*[asyncio.create_task(worker(i)) for i in range(total)])
        elapsed = time.perf_counter() - t0

    # Summary metrics
    rps = total / elapsed if elapsed else 0
    gen_tps = comp_tok / elapsed if elapsed else 0
    pre_tps = prompt_tok / elapsed if elapsed else 0
    p50 = sorted(latencies)[int(0.5 * len(latencies))] if latencies else 0
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0

    print(f"\n🚀 Completed {total} requests in {elapsed:.2f}s")
    print(f"RPS: {rps:.2f} | Prompt TPS: {pre_tps:.1f} | Gen TPS: {gen_tps:.1f}")
    print(f"Latency P50: {p50:.3f}s | P95: {p95:.3f}s")
    print(f"Tokens: prompt={prompt_tok} completion={comp_tok}")
    print(f"✅ Responses saved in: {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--json", default="/tmp/openai_messages/qwen-api_messages.json")
    ap.add_argument("--total", type=int, default=128, help="total requests")
    ap.add_argument("--concurrency", type=int, default=64, help="in-flight requests")
    ap.add_argument(
        "--randomize", action="store_true", help="add tiny random user turn"
    )
    ap.add_argument(
        "--outdir", default="blast_results", help="Directory to dump responses"
    )
    args = ap.parse_args()

    payload = load_payload(args.json)
    asyncio.run(
        run(
            args.url, payload, args.total, args.concurrency, args.randomize, args.outdir
        )
    )


# python3 blast_vllm.py   --url http://192.222.48.192:80/v1/chat/completions   --json /tmp/openai_messages/qwen-api_messages.json   --total 1024  --concurrency 128   --randomize   --outdir ./blast_runs/$(date +%Y%m%d-%H%M%S)
