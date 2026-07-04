// Intercom collector — https://developers.intercom.com/docs/references/rest-api/api.intercom.io/conversations/searchconversations
// Env: INTERCOM_ACCESS_TOKEN (EU workspaces: also set INTERCOM_API_BASE=https://api.eu.intercom.io)
const API = (process.env.INTERCOM_API_BASE ?? "https://api.intercom.io") + "/conversations/search";

const stripHtml = (s) => (s ?? "").replace(/<[^>]+>/g, "").trim();

export async function collectIntercom() {
  const token = process.env.INTERCOM_ACCESS_TOKEN;
  if (!token) {
    console.warn("[intercom] INTERCOM_ACCESS_TOKEN not set — skipping");
    return [];
  }
  const res = await fetch(API, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "Intercom-Version": "2.11",
    },
    body: JSON.stringify({
      query: { field: "state", operator: "=", value: "open" },
      pagination: { per_page: 50 },
    }),
  });
  if (!res.ok) throw new Error(`intercom ${res.status}: ${await res.text()}`);
  const { conversations } = await res.json();

  return conversations.map((c) => {
    const tags = (c.tags?.tags ?? []).map((t) => t.name);
    const sev = tags.find((t) => /^P\d Incident$/.test(t))?.split(" ")[0] ?? null;
    const email = c.source?.author?.email ?? "";
    const brand = email.match(/@(?:[\w-]+\+)?([\w-]+)\./)?.[1] ?? null;
    const body = stripHtml(c.source?.body).split("\n")[0].slice(0, 80) || "Conversation";
    return {
      id: `ic-${c.id}`,
      source: "intercom",
      title: brand ? `${body} — ${brand}` : body,
      type: "ticket",
      status: c.state === "open" ? "Open" : c.state,
      status_category: c.state === "open" ? "active" : "closed",
      severity: sev,
      brand,
      author: c.source?.author?.name ?? null,
      created_at: new Date(c.created_at * 1000).toISOString(),
      updated_at: new Date(c.updated_at * 1000).toISOString(),
      url: `https://app.eu.intercom.com/a/inbox/_/inbox/conversation/${c.id}`,
      sla_status: c.sla_applied?.sla_status ?? null,
    };
  });
}
