// Slack collector — reads #ftai-feedback via conversations.history
// Env: SLACK_BOT_TOKEN (xoxb-, scopes: channels:history, users:read)
//      SLACK_CHANNEL_ID (default: C09CYLV4XC6 = #ftai-feedback)
const CHANNEL = process.env.SLACK_CHANNEL_ID ?? "C09CYLV4XC6";

// Skip joins, bot posts, and team status updates — keep messages that produce work.
const NOISE = [/has joined the channel/, /^:\w+:.*update/i];

export async function collectSlack() {
  const token = process.env.SLACK_BOT_TOKEN;
  if (!token) {
    console.warn("[slack] SLACK_BOT_TOKEN not set — skipping");
    return [];
  }
  const oldest = (Date.now() / 1000 - 30 * 86400).toFixed(0); // last 30 days
  const res = await fetch(
    `https://slack.com/api/conversations.history?channel=${CHANNEL}&oldest=${oldest}&limit=100`,
    { headers: { Authorization: `Bearer ${token}` } }
  );
  const data = await res.json();
  if (!data.ok) throw new Error(`slack: ${data.error}`);

  return data.messages
    .filter((m) => !m.subtype && !m.bot_id && !NOISE.some((re) => re.test(m.text)))
    .map((m) => ({
      id: `slack-${m.ts}`,
      source: "slack",
      title: m.text.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").slice(0, 90),
      type: "feedback",
      status: "Open",
      status_category: "active",
      severity: null,
      brand: null,
      author: m.user ?? null, // user ID; resolve names via users.info if needed
      created_at: new Date(parseFloat(m.ts) * 1000).toISOString(),
      updated_at: new Date(parseFloat(m.ts) * 1000).toISOString(),
      url: `https://fasttrack-solutions.slack.com/archives/${CHANNEL}/p${m.ts.replace(".", "")}`,
      sla_status: null,
    }));
}
