// Fetch all three streams, normalize into WorkItems, write data/items.json.
// Usage: node collectors/run.js   (requires Node 18+, tokens via env vars)
import { writeFile } from "node:fs/promises";
import { collectSlack } from "./slack.js";
import { collectIncidentIo } from "./incidentio.js";
import { collectIntercom } from "./intercom.js";

const results = await Promise.allSettled([
  collectSlack(),
  collectIncidentIo(),
  collectIntercom(),
]);

const items = [];
for (const [i, r] of results.entries()) {
  const name = ["slack", "incident.io", "intercom"][i];
  if (r.status === "fulfilled") {
    console.log(`[${name}] ${r.value.length} items`);
    items.push(...r.value);
  } else {
    console.error(`[${name}] FAILED: ${r.reason.message}`);
  }
}

items.sort((a, b) => b.created_at.localeCompare(a.created_at));
const out = new URL("../data/items.json", import.meta.url);
await writeFile(out, JSON.stringify({ generated_at: new Date().toISOString(), items }, null, 2));
console.log(`wrote ${items.length} items → data/items.json`);
