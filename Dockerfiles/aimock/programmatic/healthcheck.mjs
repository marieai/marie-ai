import { networkInterfaces } from "node:os";

const port = process.env.AIMOCK_PORT || "4010";
const interfaces = Object.values(networkInterfaces()).flat().filter(Boolean);
const address =
  interfaces.find((entry) => entry.family === "IPv4" && !entry.internal)?.address ||
  "127.0.0.1";

try {
  const response = await fetch(`http://${address}:${port}/health`);
  process.exit(response.ok ? 0 : 1);
} catch {
  process.exit(1);
}
