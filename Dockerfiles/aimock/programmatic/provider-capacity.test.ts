import assert from "node:assert/strict";
import test from "node:test";

import { ProviderCapacity } from "./provider-capacity.js";

test("ProviderCapacity holds calls above the configured limit", async () => {
  const capacity = new ProviderCapacity(2);
  const releaseFirst = await capacity.acquire();
  const releaseSecond = await capacity.acquire();
  let thirdStarted = false;
  const third = capacity.acquire().then((release) => {
    thirdStarted = true;
    return release;
  });

  await Promise.resolve();
  assert.equal(thirdStarted, false);
  assert.deepEqual(capacity.snapshot(), {
    maxConcurrentCalls: 2,
    activeCalls: 2,
    peakConcurrentCalls: 2,
    waitingCalls: 1,
    capacityWaitCount: 1,
    capacityWaitMs: 0,
  });

  releaseFirst();
  const releaseThird = await third;
  assert.equal(thirdStarted, true);
  assert.equal(capacity.snapshot().activeCalls, 2);

  releaseSecond();
  releaseThird();
  assert.equal(capacity.snapshot().activeCalls, 0);
});

test("ProviderCapacity applies a larger limit without dropping waiters", async () => {
  const capacity = new ProviderCapacity(1);
  const releaseFirst = await capacity.acquire();
  const second = capacity.acquire();
  const third = capacity.acquire();

  capacity.setLimit(3);
  const [releaseSecond, releaseThird] = await Promise.all([second, third]);

  assert.equal(capacity.snapshot().activeCalls, 3);
  assert.equal(capacity.snapshot().peakConcurrentCalls, 3);
  releaseFirst();
  releaseSecond();
  releaseThird();
});

test("ProviderCapacity resets telemetry while preserving active calls", async () => {
  const capacity = new ProviderCapacity(1);
  const release = await capacity.acquire();

  capacity.resetTelemetry();

  assert.deepEqual(capacity.snapshot(), {
    maxConcurrentCalls: 1,
    activeCalls: 1,
    peakConcurrentCalls: 1,
    waitingCalls: 0,
    capacityWaitCount: 0,
    capacityWaitMs: 0,
  });
  release();
});

test("ProviderCapacity rejects invalid limits", () => {
  assert.throws(() => new ProviderCapacity(-1), /non-negative integer/);
  assert.throws(() => new ProviderCapacity(Number.NaN), /non-negative integer/);

  const capacity = new ProviderCapacity(4);
  assert.throws(() => capacity.setLimit(1.5), /non-negative integer/);
});
