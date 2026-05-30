import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";

export type AnnotationBox = {
  class: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type FrameAnnotation = {
  video_name: string;
  frame_id: number;
  time_sec: number;
  width: number;
  height: number;
  boxes: AnnotationBox[];
};

export type AnnotationTask = {
  id: string;
  videoName: string;
  fps: number;
  frameStep: number;
  totalFrames: number;
  width: number;
  height: number;
  startFrame: number;
  endFrame: number;
  assignee?: string;
  notes?: string;
  createdAt: string;
  updatedAt: string;
};

export type AnnotationSubmission = {
  id: string;
  taskId: string;
  annotator: string;
  videoName: string;
  fps: number;
  frameStep: number;
  totalFrames: number;
  width: number;
  height: number;
  skippedFrames: number[];
  annotations: FrameAnnotation[];
  createdAt: string;
};

export class JsonStorage {
  private root: string;

  constructor(root: string) {
    this.root = root;
  }

  async ensure() {
    await mkdir(this.dir("tasks"), { recursive: true });
    await mkdir(this.dir("submissions"), { recursive: true });
  }

  dir(name: string) {
    return path.join(this.root, name);
  }

  taskPath(id: string) {
    return path.join(this.dir("tasks"), `${id}.json`);
  }

  submissionPath(id: string) {
    return path.join(this.dir("submissions"), `${id}.json`);
  }

  async saveTask(task: AnnotationTask) {
    await this.ensure();
    await writeJson(this.taskPath(task.id), task);
  }

  async listTasks() {
    await this.ensure();
    return listJson<AnnotationTask>(this.dir("tasks"));
  }

  async getTask(id: string) {
    return readJson<AnnotationTask>(this.taskPath(id));
  }

  async saveSubmission(submission: AnnotationSubmission) {
    await this.ensure();
    await writeJson(this.submissionPath(submission.id), submission);
  }

  async listSubmissions(taskId?: string) {
    await this.ensure();
    const submissions = await listJson<AnnotationSubmission>(this.dir("submissions"));
    return taskId ? submissions.filter((item) => item.taskId === taskId) : submissions;
  }
}

async function writeJson(filename: string, value: unknown) {
  await writeFile(filename, `${JSON.stringify(value, null, 2)}\n`);
}

async function readJson<T>(filename: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(filename, "utf8")) as T;
  } catch (error: any) {
    if (error?.code === "ENOENT") return null;
    throw error;
  }
}

async function listJson<T>(dir: string): Promise<T[]> {
  const files = await readdir(dir);
  const items = await Promise.all(
    files
      .filter((file) => file.endsWith(".json"))
      .map((file) => readJson<T>(path.join(dir, file)))
  );
  return items.filter((item): item is T => Boolean(item));
}
