// incident.io collector — https://api-docs.incident.io/tag/Incidents-V2
// Env: INCIDENT_IO_API_KEY
const API = "https://api.incident.io/v2/incidents";

export async function collectIncidentIo() {
  const key = process.env.INCIDENT_IO_API_KEY;
  if (!key) {
    console.warn("[incident.io] INCIDENT_IO_API_KEY not set — skipping");
    return [];
  }
  const res = await fetch(`${API}?page_size=50`, {
    headers: { Authorization: `Bearer ${key}` },
  });
  if (!res.ok) throw new Error(`incident.io ${res.status}: ${await res.text()}`);
  const { incidents } = await res.json();

  return incidents.map((inc) => ({
    id: inc.reference?.toLowerCase() ?? inc.id,
    source: "incident_io",
    title: inc.name,
    type: "incident",
    status: inc.incident_status?.name ?? "Unknown",
    status_category: inc.incident_status?.category ?? "active",
    severity: inc.severity?.name ?? null,
    brand: null,
    author: null,
    created_at: inc.created_at,
    updated_at: inc.updated_at ?? inc.created_at,
    url: inc.permalink,
    sla_status: null,
  }));
}
