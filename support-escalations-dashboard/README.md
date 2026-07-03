# Support Escalations Dashboard (POC)

One view over the three streams that produce support work for FastTrack AI:

| Stream | What it carries | API used |
|--------|-----------------|----------|
| Slack `#ftai-feedback` | Feedback / requests from partner managers | `conversations.history` |
| Incident.io | Incidents (alerts, queue buildups, Intercom-escalated) | `GET /v2/incidents` |
| Intercom | Client-raised tickets ("something's not working") | `POST /conversations/search` |

Everything is normalized into one **WorkItem** schema so the dashboard can show a
unified queue, severity/SLA state, and per-stream inflow.

```mermaid
flowchart LR
    S[Slack #ftai-feedback] -->|conversations.history| N
    I[Incident.io] -->|/v2/incidents| N
    C[Intercom] -->|/conversations/search| N
    N[collectors/run.js<br/>normalize → WorkItem] --> J[(data/items.json)]
    M[Claude session via MCPs] -.->|alternative refresh path| J
    J --> D[dashboard/index.html<br/>static, no build step]
```

## Quick start (sample data, no tokens needed)

```bash
cd support-escalations-dashboard
python3 -m http.server 8080
# open http://localhost:8080/dashboard/
```

`data/sample-items.json` is a real snapshot (47 items) pulled 2026-07-03 from all
three sources via MCPs. Customer email addresses were stripped; brand names kept.

## Live data (API mode)

```bash
export SLACK_BOT_TOKEN=xoxb-...          # scopes: channels:history
export INCIDENT_IO_API_KEY=...
export INTERCOM_ACCESS_TOKEN=...         # EU workspace: also INTERCOM_API_BASE=https://api.eu.intercom.io
node collectors/run.js                   # writes data/items.json (Node 18+)
```

The dashboard prefers `data/items.json` and falls back to the sample. A collector
with a missing token is skipped, so partial refreshes work.

## Live data (MCP mode)

In a Claude session with the Slack / incident.io / Intercom MCPs connected, ask
Claude to refresh `data/items.json` in the WorkItem schema — that's how the
sample snapshot was produced. No tokens leave the MCP layer.

## WorkItem schema

```json
{
  "id": "inc-12204",
  "source": "slack | incident_io | intercom",
  "title": "llm-agent transient 'socket hang up' — supabets",
  "type": "incident | ticket | feedback | bug | request | question",
  "status": "Open | Triage | Investigating | Monitoring | Closed | ...",
  "status_category": "triage | active | closed",
  "severity": "P1 | P2 | P3 | null",
  "brand": "supabets | null",
  "author": "reporter name | null",
  "created_at": "2026-07-03T13:34:55Z",
  "updated_at": "2026-07-03T13:36:09Z",
  "url": "deep link back to the source system",
  "sla_status": "missed | hit | active | null   (Intercom only)"
}
```

## What the dashboard shows

- **Stat tiles** — open items, SLA-missed count, P1/P2 open, oldest open item
- **Open work by stream** — where the load is right now
- **Open by severity** — P1/P2/P3 vs unclassified
- **Inflow (21 days)** — daily new items stacked by stream
- **Unified work-item table** — open first, newest first, deep links to source

Stream filter chips at the top apply to every tile, chart, and the table.

## POC limitations (deliberate)

- No persistence/history — each refresh overwrites `items.json`; trend lines
  beyond created_at need a store (SQLite would do).
- Slack items are never "closed" — there's no done-signal in a channel. Real
  version needs a convention (✅ reaction) or a triage step.
- Slack collector returns user IDs, not names (needs `users.info` lookup).
- No dedup across streams — an Intercom ticket escalated to Incident.io shows
  twice (INC-12206 ↔ its Intercom conversation). Real version should link them
  via the Intercom URL already present in incident summaries.
- Severity for Slack items is unknowable without classification — an LLM pass
  could assign type/severity/brand from message text.
