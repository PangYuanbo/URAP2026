# Annotation Monorepo Deployment

This repo now contains a small annotation web app and API:

```text
apps/web   Cloudflare Pages frontend
apps/api   Railway Express API
```

The frontend still uses local video files in the browser. Annotators select the shared video on their own machine, draw boxes, and submit JSON annotations to the Railway API.

## Railway API

Create a Railway service from this repo.

Recommended Railway settings:

- Root directory: repo root
- Build command: `npm install && npm run build -w apps/api`
- Start command: `npm run start -w apps/api`
- Healthcheck path: `/health`

Environment variables:

```text
DATA_DIR=/data/annotation-api
FRONTEND_ORIGIN=https://YOUR-CLOUDFLARE-PAGES-DOMAIN.pages.dev
API_TOKEN=choose-a-private-admin-token
```

Attach a Railway volume mounted at `/data` if you want submissions to persist across deploys.

## Cloudflare Pages

Create a Pages project from this repo.

Settings:

- Build command: `npm install && npm run build -w apps/web`
- Build output directory: `apps/web/dist`

After deploy, open the Cloudflare URL, enter the Railway API URL, the task id, and the annotator name.

## Create Annotation Tasks

Task creation is protected by `API_TOKEN` when it is set.

Example:

```bash
curl -X POST "$API_URL/api/tasks" \
  -H "authorization: Bearer $API_TOKEN" \
  -H "content-type: application/json" \
  -d '{
    "videoName": "dji_fly_20260522_113924_10_1779475848691_hdrvideo.MP4",
    "fps": 29.97,
    "frameStep": 5,
    "totalFrames": 8718,
    "width": 3840,
    "height": 2160,
    "startFrame": 40,
    "endFrame": 2000,
    "assignee": "alice",
    "notes": "522 video segment 1"
  }'
```

Make one task per annotator/range. Give each annotator:

- The Cloudflare Pages URL
- The video file
- Their task id
- Their start/end range if you want them to enter it manually too

## Export Merged Results

Download merged annotations for a task:

```bash
curl "$API_URL/api/export/TASK_ID" -o merged_task.json
```

If multiple submissions include the same frame, the latest submission wins in export order.
