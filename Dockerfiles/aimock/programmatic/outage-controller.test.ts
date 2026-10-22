import assert from "node:assert/strict";
import test from "node:test";

import { beginTimedOutage } from "./outage-controller.js";

test("beginTimedOutage stops the inference listener before scheduling recovery", async () => {
  const events: string[] = [];
  const recovered = new Promise<void>((resolve) => {
    void beginTimedOutage({
      durationMs: 5,
      stop: async () => {
        events.push("stopped");
      },
      start: async () => {
        events.push("started");
        resolve();
      },
      onRecoveryError: (error) => {
        throw error;
      },
    }).then(() => {
      assert.deepEqual(events, ["stopped"]);
    });
  });

  await recovered;
  assert.deepEqual(events, ["stopped", "started"]);
});
