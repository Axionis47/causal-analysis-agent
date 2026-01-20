# SSE Streaming Guide

## Overview

The analysis progress endpoint streams events over SSE:

- Endpoint: `GET /api/v1/analyses/{id}/stream`
- Primary transport: Redis pubsub for real-time updates
- Fallback: database polling when Redis is unavailable

## Event Types

Progress events use a JSON payload with these fields:

- `analysis_id`: Analysis UUID as a string
- `stage`: Pipeline stage name
- `status`: Stage status (`running`, `completed`, `failed`, `cancelled`)
- `progress_percent`: Integer 0-100
- `message`: Human-readable status
- `error`: Error message (nullable)

Additional stream messages:

- `disconnected`: JSON payload with `analysis_id` and `reason`
  (`timeout`, `error`, `completed`, `stream_end`)
- `: heartbeat`: SSE comment sent every 30 seconds to keep the connection alive

Example SSE payload:

```text
event: progress
data: {"analysis_id":"...","stage":"eda","status":"running","progress_percent":35,"message":"Analyzing columns","error":null}
```

## Timeout Behavior

Streams time out after `PROGRESS_STREAM_TIMEOUT` seconds. Clients should reconnect with exponential backoff and stop retrying after repeated failures or when the analysis status is terminal.

## Client Reconnection (TypeScript)

```ts
const baseDelayMs = 500;
const maxDelayMs = 10_000;
let retries = 0;
let source: EventSource | null = null;

const connect = () => {
  const url = `/api/v1/analyses/${analysisId}/stream`;
  source = new EventSource(url);

  source.addEventListener("progress", (event) => {
    retries = 0;
    const data = JSON.parse((event as MessageEvent).data);
    console.log("progress", data);
  });

  source.addEventListener("disconnected", (event) => {
    const data = JSON.parse((event as MessageEvent).data);
    console.warn("disconnected", data);
  });

  source.onerror = () => {
    source?.close();
    const delay = Math.min(baseDelayMs * 2 ** retries, maxDelayMs);
    retries += 1;
    window.setTimeout(connect, delay);
  };
};

connect();
```

## Best Practices

- Treat `completed`, `failed`, and `cancelled` statuses as terminal.
- Reconnect with exponential backoff; cap retry attempts to avoid thundering herds.
- If the stream ends unexpectedly, call `GET /api/v1/analyses/{id}` to confirm status.
- Heartbeats are SSE comments for keepalive; rely on `disconnected` and `onerror`
  callbacks to detect drops.
