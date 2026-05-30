import cors from "cors";
import express from "express";
import { nanoid } from "nanoid";
import { z } from "zod";
import { JsonStorage, type AnnotationSubmission } from "./storage.js";

const port = Number(process.env.PORT || 3000);
const dataDir = process.env.DATA_DIR || "./annotation-data";
const frontendOrigin = process.env.FRONTEND_ORIGIN || "*";
const apiToken = process.env.API_TOKEN || "";

const storage = new JsonStorage(dataDir);
const app = express();

app.use(cors({ origin: frontendOrigin === "*" ? true : frontendOrigin }));
app.use(express.json({ limit: "50mb" }));

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.get("/api/tasks", async (_req, res, next) => {
  try {
    const tasks = await storage.listTasks();
    res.json(tasks.sort((a, b) => b.createdAt.localeCompare(a.createdAt)));
  } catch (error) {
    next(error);
  }
});

app.post("/api/tasks", requireToken, async (req, res, next) => {
  try {
    const input = taskSchema.parse(req.body);
    const now = new Date().toISOString();
    const task = {
      id: nanoid(12),
      ...input,
      createdAt: now,
      updatedAt: now
    };
    await storage.saveTask(task);
    res.status(201).json(task);
  } catch (error) {
    next(error);
  }
});

app.get("/api/tasks/:id", async (req, res, next) => {
  try {
    const task = await storage.getTask(req.params.id);
    if (!task) return res.status(404).json({ error: "Task not found" });
    res.json(task);
  } catch (error) {
    next(error);
  }
});

app.get("/api/submissions", async (req, res, next) => {
  try {
    const submissions = await storage.listSubmissions(req.query.taskId?.toString());
    res.json(submissions.sort((a, b) => b.createdAt.localeCompare(a.createdAt)));
  } catch (error) {
    next(error);
  }
});

app.post("/api/submissions", async (req, res, next) => {
  try {
    const input = submissionSchema.parse(req.body);
    const task = await storage.getTask(input.taskId);
    if (!task) return res.status(404).json({ error: "Task not found" });

    const submission: AnnotationSubmission = {
      id: nanoid(12),
      taskId: input.taskId,
      annotator: input.annotator,
      videoName: input.videoName,
      fps: input.fps,
      frameStep: input.frameStep,
      totalFrames: input.totalFrames,
      width: input.width,
      height: input.height,
      skippedFrames: input.skippedFrames ?? [],
      annotations: input.annotations,
      createdAt: new Date().toISOString()
    };
    await storage.saveSubmission(submission);
    res.status(201).json({ id: submission.id, annotations: submission.annotations.length });
  } catch (error) {
    next(error);
  }
});

app.get("/api/export/:taskId", async (req, res, next) => {
  try {
    const task = await storage.getTask(req.params.taskId);
    if (!task) return res.status(404).json({ error: "Task not found" });

    const submissions = await storage.listSubmissions(task.id);
    const byFrame = new Map<number, any>();
    const skipped = new Set<number>();
    for (const submission of submissions.sort((a, b) => a.createdAt.localeCompare(b.createdAt))) {
      for (const frame of submission.skippedFrames) skipped.add(frame);
      for (const annotation of submission.annotations) {
        byFrame.set(annotation.frame_id, {
          ...annotation,
          source_submission_id: submission.id,
          annotator: submission.annotator
        });
      }
    }

    res.json({
      version: 1,
      video_name: task.videoName,
      fps: task.fps,
      frame_step: task.frameStep,
      assignment: {
        start_frame: task.startFrame,
        end_frame: task.endFrame
      },
      total_frames: task.totalFrames,
      width: task.width,
      height: task.height,
      merged_from: submissions.map((item) => ({
        id: item.id,
        annotator: item.annotator,
        createdAt: item.createdAt,
        annotations: item.annotations.length
      })),
      skipped_frames: [...skipped].sort((a, b) => a - b),
      annotations: [...byFrame.values()].sort((a, b) => a.frame_id - b.frame_id)
    });
  } catch (error) {
    next(error);
  }
});

app.use((error: any, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  if (error instanceof z.ZodError) {
    return res.status(400).json({ error: "Invalid request", details: error.flatten() });
  }
  console.error(error);
  res.status(500).json({ error: "Internal server error" });
});

storage.ensure().then(() => {
  app.listen(port, () => {
    console.log(`annotation api listening on ${port}`);
  });
});

function requireToken(req: express.Request, res: express.Response, next: express.NextFunction) {
  if (!apiToken) return next();
  const header = req.header("authorization") || "";
  if (header === `Bearer ${apiToken}`) return next();
  res.status(401).json({ error: "Unauthorized" });
}

const taskSchema = z.object({
  videoName: z.string().min(1),
  fps: z.number().positive(),
  frameStep: z.number().int().positive(),
  totalFrames: z.number().int().positive(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  startFrame: z.number().int().min(0),
  endFrame: z.number().int().min(0),
  assignee: z.string().optional(),
  notes: z.string().optional()
});

const boxSchema = z.object({
  class: z.string().min(1),
  x1: z.number(),
  y1: z.number(),
  x2: z.number(),
  y2: z.number()
});

const annotationSchema = z.object({
  video_name: z.string().min(1),
  frame_id: z.number().int().min(0),
  time_sec: z.number().min(0),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  boxes: z.array(boxSchema)
});

const submissionSchema = z.object({
  taskId: z.string().min(1),
  annotator: z.string().min(1),
  videoName: z.string().min(1),
  fps: z.number().positive(),
  frameStep: z.number().int().positive(),
  totalFrames: z.number().int().positive(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  skippedFrames: z.array(z.number().int().min(0)).optional(),
  annotations: z.array(annotationSchema)
});
