export interface TimedOutageOptions {
  durationMs: number;
  stop: () => Promise<void>;
  start: () => Promise<void>;
  onRecoveryError: (error: unknown) => void;
}

export async function beginTimedOutage(
  options: TimedOutageOptions,
): Promise<ReturnType<typeof setTimeout>> {
  await options.stop();
  return setTimeout(() => {
    void options.start().catch(options.onRecoveryError);
  }, options.durationMs);
}
