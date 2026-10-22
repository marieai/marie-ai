type Release = () => void;

type Waiter = {
  startedAt: number;
  resolve: (release: Release) => void;
};

export type ProviderCapacitySnapshot = {
  maxConcurrentCalls: number;
  activeCalls: number;
  peakConcurrentCalls: number;
  waitingCalls: number;
  capacityWaitCount: number;
  capacityWaitMs: number;
};

export class ProviderCapacity {
  private maxConcurrentCalls: number;
  private activeCalls = 0;
  private peakConcurrentCalls = 0;
  private capacityWaitCount = 0;
  private capacityWaitMs = 0;
  private readonly waiters: Waiter[] = [];

  constructor(maxConcurrentCalls: number) {
    this.assertLimit(maxConcurrentCalls);
    this.maxConcurrentCalls = maxConcurrentCalls;
  }

  acquire(): Promise<Release> {
    if (this.hasRoom()) {
      return Promise.resolve(this.startCall());
    }

    this.capacityWaitCount += 1;
    return new Promise((resolve) => {
      this.waiters.push({ startedAt: Date.now(), resolve });
    });
  }

  setLimit(limit: number): void {
    this.assertLimit(limit);
    this.maxConcurrentCalls = limit;
    this.drain();
  }

  resetTelemetry(): void {
    this.peakConcurrentCalls = this.activeCalls;
    this.capacityWaitCount = 0;
    this.capacityWaitMs = 0;
  }

  snapshot(): ProviderCapacitySnapshot {
    return {
      maxConcurrentCalls: this.maxConcurrentCalls,
      activeCalls: this.activeCalls,
      peakConcurrentCalls: this.peakConcurrentCalls,
      waitingCalls: this.waiters.length,
      capacityWaitCount: this.capacityWaitCount,
      capacityWaitMs: this.capacityWaitMs,
    };
  }

  private hasRoom(): boolean {
    return this.maxConcurrentCalls === 0 || this.activeCalls < this.maxConcurrentCalls;
  }

  private assertLimit(limit: number): void {
    if (!Number.isSafeInteger(limit) || limit < 0) {
      throw new Error("Provider concurrency limit must be a non-negative integer");
    }
  }

  private startCall(): Release {
    this.activeCalls += 1;
    this.peakConcurrentCalls = Math.max(this.peakConcurrentCalls, this.activeCalls);
    let released = false;
    return () => {
      if (released) {
        return;
      }
      released = true;
      this.activeCalls -= 1;
      this.drain();
    };
  }

  private drain(): void {
    while (this.waiters.length > 0 && this.hasRoom()) {
      const waiter = this.waiters.shift();
      if (!waiter) {
        return;
      }
      this.capacityWaitMs += Date.now() - waiter.startedAt;
      waiter.resolve(this.startCall());
    }
  }
}
