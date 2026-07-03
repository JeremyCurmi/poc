# No build step (zero-dep Node server + static dashboard), so a single slim stage
# is the whole image. All config via env vars; no secrets baked in.
FROM node:22-alpine
WORKDIR /app
COPY server.js ./
COPY collectors ./collectors
COPY dashboard ./dashboard
COPY data/sample-items.json ./data/sample-items.json
ENV PORT=8080
EXPOSE 8080
# Mount a volume at /app/data to persist items.json + triage-state.json
CMD ["node", "server.js"]
