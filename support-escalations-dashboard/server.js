// Support escalations dashboard — actionable server (Node 18+, zero deps).
//
//   node server.js
//
// Serves the dashboard and adds a triage layer on top of the read-only streams:
//   GET  /api/items    → work items merged with triage state + capabilities
//   POST /api/action   → { id, action: ack|snooze|done|reopen, assignee?, note?, hours? }
//   POST /api/linear   → { id } — create a Linear issue from a work item
//   POST /api/refresh  → re-run the collectors (writes data/items.json)
//
// Env: PORT (default 8080)
//      BASIC_AUTH=user:pass         optional, protects everything when deployed
//      LINEAR_API_KEY, LINEAR_TEAM_ID   enables the → Linear action
//      + collector tokens (see collectors/*.js) for /api/refresh
import { createServer } from "node:http";
import { readFile, writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { extname, normalize, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL(".", import.meta.url));
const PORT = Number(process.env.PORT ?? 8080);
const STATE_FILE = join(ROOT, "data", "triage-state.json");

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml" };

async function readJson(path, fallback) {
  try { return JSON.parse(await readFile(path, "utf8")); } catch { return fallback; }
}

async function loadItems() {
  const data =
    (await readJson(join(ROOT, "data", "items.json"), null)) ??
    (await readJson(join(ROOT, "data", "sample-items.json"), { generated_at: null, items: [] }));
  const triage = await readJson(STATE_FILE, {});
  for (const it of data.items) it.triage = triage[it.id] ?? null;
  return data;
}

async function saveTriage(id, patch) {
  const triage = await readJson(STATE_FILE, {});
  if (patch === null) delete triage[id];
  else triage[id] = { ...triage[id], ...patch, updated_at: new Date().toISOString() };
  await writeFile(STATE_FILE, JSON.stringify(triage, null, 2));
  return triage[id] ?? null;
}

async function createLinearIssue(item) {
  const key = process.env.LINEAR_API_KEY, team = process.env.LINEAR_TEAM_ID;
  if (!key || !team) throw new Error("LINEAR_API_KEY / LINEAR_TEAM_ID not set");
  const res = await fetch("https://api.linear.app/graphql", {
    method: "POST",
    headers: { Authorization: key, "Content-Type": "application/json" },
    body: JSON.stringify({
      query: `mutation($input: IssueCreateInput!) {
        issueCreate(input: $input) { issue { url identifier } } }`,
      variables: {
        input: {
          teamId: team,
          title: `[${item.source}] ${item.title}`,
          description: [
            `Escalated from the support dashboard.`,
            `- Source: ${item.source}`,
            item.severity && `- Severity: ${item.severity}`,
            item.brand && `- Brand: ${item.brand}`,
            item.author && `- Reporter: ${item.author}`,
            `- Created: ${item.created_at}`,
            `- Link: ${item.url}`,
          ].filter(Boolean).join("\n"),
        },
      },
    }),
  });
  const body = await res.json();
  const issue = body.data?.issueCreate?.issue;
  if (!issue) throw new Error(`linear: ${JSON.stringify(body.errors ?? body)}`);
  return issue;
}

const readBody = (req) => new Promise((resolve, reject) => {
  let buf = "";
  req.on("data", (c) => { buf += c; if (buf.length > 65536) req.destroy(); });
  req.on("end", () => { try { resolve(JSON.parse(buf || "{}")); } catch (e) { reject(e); } });
});

const json = (res, code, obj) =>
  res.writeHead(code, { "Content-Type": "application/json" }).end(JSON.stringify(obj));

const server = createServer(async (req, res) => {
  if (process.env.BASIC_AUTH) {
    const expected = "Basic " + Buffer.from(process.env.BASIC_AUTH).toString("base64");
    if (req.headers.authorization !== expected) {
      return res.writeHead(401, { "WWW-Authenticate": 'Basic realm="dashboard"' }).end();
    }
  }
  const url = new URL(req.url, "http://x");
  try {
    if (url.pathname === "/api/items") {
      const data = await loadItems();
      return json(res, 200, {
        ...data,
        capabilities: { linear: Boolean(process.env.LINEAR_API_KEY && process.env.LINEAR_TEAM_ID) },
      });
    }
    if (url.pathname === "/api/action" && req.method === "POST") {
      const { id, action, assignee, note, hours } = await readBody(req);
      if (!id || !action) return json(res, 400, { error: "id and action required" });
      const patch = {
        ack:    { state: "acked", ...(assignee && { assignee }), ...(note && { note }) },
        snooze: { state: "snoozed", snoozed_until: new Date(Date.now() + (hours ?? 24) * 3600e3).toISOString() },
        done:   { state: "done" },
        reopen: null,
      }[action];
      if (patch === undefined) return json(res, 400, { error: `unknown action: ${action}` });
      return json(res, 200, { id, triage: await saveTriage(id, patch) });
    }
    if (url.pathname === "/api/linear" && req.method === "POST") {
      const { id } = await readBody(req);
      const { items } = await loadItems();
      const item = items.find((it) => it.id === id);
      if (!item) return json(res, 404, { error: "unknown item" });
      const issue = await createLinearIssue(item);
      const triage = await saveTriage(id, { state: "acked", linear_url: issue.url, linear_id: issue.identifier });
      return json(res, 200, { id, issue, triage });
    }
    if (url.pathname === "/api/refresh" && req.method === "POST") {
      await new Promise((resolve, reject) =>
        execFile("node", [join(ROOT, "collectors", "run.js")], (err, stdout, stderr) =>
          err ? reject(new Error(stderr || err.message)) : resolve(stdout)));
      return json(res, 200, { ok: true });
    }

    // static files
    let path = url.pathname === "/" ? "/dashboard/index.html" : url.pathname;
    if (path === "/dashboard/" || path === "/dashboard") path = "/dashboard/index.html";
    const file = normalize(join(ROOT, path));
    if (!file.startsWith(ROOT)) return res.writeHead(403).end();
    try {
      const body = await readFile(file);
      res.writeHead(200, { "Content-Type": MIME[extname(file)] ?? "application/octet-stream" }).end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  } catch (e) {
    json(res, 500, { error: e.message });
  }
});

server.listen(PORT, () => console.log(`dashboard on http://localhost:${PORT}/`));
