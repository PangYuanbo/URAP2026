from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from qstr_dronedet.tracking.action_chunk import (
    actions_from_boxes,
    attach_frame_priors_to_tracklets,
    build_frame_prior_index_from_heatmaps,
    build_action_chunk_samples_from_rows,
    export_action_chunk_dataset_from_tracklets,
    export_action_prior_heatmaps_from_sample_scores,
    merge_action_chunk_datasets,
    reconstruct_boxes,
    split_action_chunk_dataset,
)
from qstr_dronedet.tracking.proposal_tracklets import build_proposal_tracklet_dataset


@dataclass(frozen=True)
class ActionPolicyEvalResult:
    out_path: Path
    summary: dict[str, Any]


class ActionChunkMLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionChunkDiffusionMLP(torch.nn.Module):
    def __init__(self, cond_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cond_dim + action_dim + 1, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, action_dim),
        )

    def forward(self, cond: torch.Tensor, noisy_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(1)
        return self.net(torch.cat([cond, noisy_action, timestep], dim=1))


def _load_action_samples(jsonl_path: str | Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with Path(jsonl_path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _sample_to_arrays(sample: dict[str, Any], target_mode: str = "direct") -> tuple[np.ndarray, np.ndarray]:
    past_boxes = np.asarray(sample["past_boxes"], dtype=np.float32)
    past_scores = np.asarray(sample.get("past_scores", []), dtype=np.float32).reshape(-1, 1)
    past_visible = np.asarray(sample.get("past_visible", []), dtype=np.float32).reshape(-1, 1)
    future_actions = np.asarray(sample["future_actions"], dtype=np.float32)
    if past_scores.shape[0] != past_boxes.shape[0]:
        past_scores = np.ones((past_boxes.shape[0], 1), dtype=np.float32)
    if past_visible.shape[0] != past_boxes.shape[0]:
        past_visible = np.ones((past_boxes.shape[0], 1), dtype=np.float32)
    x = np.concatenate([past_boxes, past_scores, past_visible], axis=1).reshape(-1)
    if target_mode == "residual_cv":
        future_actions = future_actions - _constant_velocity_actions(past_boxes, len(future_actions))
    elif target_mode != "direct":
        raise ValueError(f"unknown action target mode: {target_mode}")
    y = future_actions.reshape(-1)
    return x.astype(np.float32), y.astype(np.float32)


def _samples_to_tensors(samples: list[dict[str, Any]], target_mode: str = "direct") -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for sample in samples:
        x, y = _sample_to_arrays(sample, target_mode=target_mode)
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError("Action-chunk dataset is empty")
    return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(np.stack(ys), dtype=torch.float32)


def _balanced_sample_weights(samples: list[dict[str, Any]], balance_by: list[str] | None) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not balance_by:
        return None, {"enabled": False, "balance_by": []}
    groups: list[str] = []
    counts: dict[str, int] = {}
    for sample in samples:
        key = "|".join(str(sample.get(field, "")) for field in balance_by)
        groups.append(key)
        counts[key] = counts.get(key, 0) + 1
    weights = torch.tensor([1.0 / max(1, counts[group]) for group in groups], dtype=torch.double)
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights, {
        "enabled": True,
        "balance_by": list(balance_by),
        "group_counts": counts,
        "min_weight": float(weights.min().item()) if len(weights) else 0.0,
        "max_weight": float(weights.max().item()) if len(weights) else 0.0,
    }


def _make_loader(
    x_norm: torch.Tensor,
    y_norm: torch.Tensor,
    samples: list[dict[str, Any]],
    batch_size: int,
    balance_by: list[str] | None,
) -> tuple[DataLoader, dict[str, Any]]:
    sample_weights, balance_summary = _balanced_sample_weights(samples, balance_by)
    sampler = None
    shuffle = True
    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(samples), replacement=True)
        shuffle = False
    loader = DataLoader(
        TensorDataset(x_norm, y_norm),
        batch_size=min(batch_size, len(samples)),
        shuffle=shuffle,
        sampler=sampler,
    )
    return loader, balance_summary


def _diffusion_schedule(steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if steps <= 1:
        raise ValueError("diffusion_steps must be greater than 1")
    betas = torch.linspace(1e-4, 2e-2, steps, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def train_action_chunk_policy(
    jsonl_path: str | Path,
    out: str | Path,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 64,
    balance_by: list[str] | None = None,
    model_type: str = "mlp",
    diffusion_steps: int = 16,
) -> Path:
    samples = _load_action_samples(jsonl_path)
    if model_type not in {"mlp", "diffusion", "residual_mlp"}:
        raise ValueError("model_type must be 'mlp', 'diffusion', or 'residual_mlp'")
    target_mode = "residual_cv" if model_type == "residual_mlp" else "direct"
    x, y = _samples_to_tensors(samples, target_mode=target_mode)
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1e-6)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0).clamp_min(1e-6)
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    loader, balance_summary = _make_loader(x_norm, y_norm, samples, batch_size, balance_by)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "diffusion":
        model = ActionChunkDiffusionMLP(cond_dim=x.shape[1], action_dim=y.shape[1], hidden=hidden).to(device)
        _, _, alpha_bars_cpu = _diffusion_schedule(diffusion_steps)
        alpha_bars = alpha_bars_cpu.to(device)
    else:
        model = ActionChunkMLP(in_dim=x.shape[1], out_dim=y.shape[1], hidden=hidden).to(device)
        alpha_bars_cpu = torch.empty(0, dtype=torch.float32)
        alpha_bars = alpha_bars_cpu.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            if model_type == "diffusion":
                t_idx = torch.randint(0, diffusion_steps, (int(by.shape[0]),), device=device)
                noise = torch.randn_like(by)
                alpha_bar = alpha_bars[t_idx].unsqueeze(1)
                noisy_action = torch.sqrt(alpha_bar) * by + torch.sqrt(1.0 - alpha_bar) * noise
                t_norm = t_idx.float().unsqueeze(1) / float(diffusion_steps - 1)
                loss = loss_fn(model(bx, noisy_action, t_norm), noise)
            else:
                loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(bx.shape[0])
            total += int(bx.shape[0])
        history.append({"epoch": epoch + 1, "loss": total_loss / max(1, total)})
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "in_dim": int(x.shape[1]),
            "out_dim": int(y.shape[1]),
            "hidden": hidden,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "history": history,
            "num_samples": len(samples),
            "balance": balance_summary,
            "model_type": model_type,
            "diffusion_steps": int(diffusion_steps) if model_type == "diffusion" else 0,
            "diffusion_alpha_bars": alpha_bars_cpu,
        },
        out_path,
    )
    return out_path


def _load_policy(weights: str | Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(weights, map_location="cpu")
    model_type = str(ckpt.get("model_type", "mlp"))
    if model_type == "diffusion":
        model = ActionChunkDiffusionMLP(cond_dim=int(ckpt["in_dim"]), action_dim=int(ckpt["out_dim"]), hidden=int(ckpt["hidden"]))
    else:
        model = ActionChunkMLP(in_dim=int(ckpt["in_dim"]), out_dim=int(ckpt["out_dim"]), hidden=int(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _predict_actions(model: torch.nn.Module, ckpt: dict[str, Any], x_norm: torch.Tensor) -> torch.Tensor:
    model_type = str(ckpt.get("model_type", "mlp"))
    if model_type != "diffusion":
        return model(x_norm)

    steps = int(ckpt["diffusion_steps"])
    _, _, alpha_bars = _diffusion_schedule(steps)
    action = torch.zeros((x_norm.shape[0], int(ckpt["out_dim"])), dtype=torch.float32)
    for step in reversed(range(steps)):
        alpha_bar = alpha_bars[step]
        t_norm = torch.full((x_norm.shape[0], 1), float(step) / float(steps - 1), dtype=torch.float32)
        eps = model(x_norm, action, t_norm)
        pred_clean = (action - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        if step > 0:
            prev_alpha_bar = alpha_bars[step - 1]
            action = torch.sqrt(prev_alpha_bar) * pred_clean + torch.sqrt(1.0 - prev_alpha_bar) * eps
        else:
            action = pred_clean
    return action


def _center_error(pred_boxes: np.ndarray, target_boxes: np.ndarray) -> float:
    if len(pred_boxes) == 0:
        return 0.0
    pred = np.asarray(pred_boxes, dtype=np.float32)
    target = np.asarray(target_boxes, dtype=np.float32)
    pred_centers = np.column_stack(((pred[:, 0] + pred[:, 2]) / 2.0, (pred[:, 1] + pred[:, 3]) / 2.0))
    target_centers = np.column_stack(((target[:, 0] + target[:, 2]) / 2.0, (target[:, 1] + target[:, 3]) / 2.0))
    return float(np.mean(np.linalg.norm(pred_centers - target_centers, axis=1)))


def _stable_sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0:
        z = np.exp(-min(value, 60.0))
        return float(1.0 / (1.0 + z))
    z = np.exp(max(value, -60.0))
    return float(z / (1.0 + z))


def _constant_velocity_actions(past_boxes: np.ndarray, future_len: int) -> np.ndarray:
    boxes = np.asarray(past_boxes, dtype=np.float32)
    if len(boxes) >= 2:
        step = actions_from_boxes(boxes[-2:])[0]
    else:
        step = np.zeros((4,), dtype=np.float32)
    return np.tile(step.reshape(1, 4), (future_len, 1)).astype(np.float32)


def _predict_sample_errors(
    sample: dict[str, Any],
    model: torch.nn.Module,
    ckpt: dict[str, Any],
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, _ = _sample_to_arrays(sample)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    x_norm = (x_tensor - ckpt["x_mean"]) / ckpt["x_std"]
    pred_actions = (_predict_actions(model, ckpt, x_norm).squeeze(0) * ckpt["y_std"] + ckpt["y_mean"]).numpy().reshape(-1, 4)
    past_boxes = np.asarray(sample["past_boxes"], dtype=np.float32)
    target_boxes = np.asarray(sample["future_boxes"], dtype=np.float32)
    cv_actions = _constant_velocity_actions(past_boxes, len(target_boxes))
    if str(ckpt.get("model_type", "mlp")) == "residual_mlp":
        pred_actions = cv_actions + pred_actions
    learned_boxes = reconstruct_boxes(past_boxes[-1], pred_actions)
    cv_boxes = reconstruct_boxes(past_boxes[-1], cv_actions)
    return _center_error(learned_boxes, target_boxes), _center_error(cv_boxes, target_boxes), pred_actions, cv_actions, learned_boxes, cv_boxes


def evaluate_action_chunk_policy(jsonl_path: str | Path, weights: str | Path, out: str | Path) -> ActionPolicyEvalResult:
    samples = _load_action_samples(jsonl_path)
    if not samples:
        raise ValueError("Action-chunk dataset is empty")
    model, ckpt = _load_policy(weights)
    rows = []
    learned_errors = []
    cv_errors = []
    with torch.no_grad():
        for sample in samples:
            learned_error, cv_error, pred_actions, cv_actions, learned_boxes, cv_boxes = _predict_sample_errors(sample, model, ckpt)
            learned_errors.append(learned_error)
            cv_errors.append(cv_error)
            rows.append(
                {
                    "seq": sample.get("seq", ""),
                    "track_id": sample.get("track_id", ""),
                    "anchor_frame": sample.get("anchor_frame", 0),
                    "label": int(float(sample.get("label", 0))),
                    "learned_center_error": learned_error,
                    "constant_velocity_center_error": cv_error,
                    "learned_actions": pred_actions.tolist(),
                    "constant_velocity_actions": cv_actions.tolist(),
                    "learned_boxes": learned_boxes.tolist(),
                    "constant_velocity_boxes": cv_boxes.tolist(),
                    "future_boxes": sample.get("future_boxes", []),
                }
            )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "jsonl": str(jsonl_path),
        "weights": str(weights),
        "out": str(out_path),
        "model_type": str(ckpt.get("model_type", "mlp")),
        "diffusion_steps": int(ckpt.get("diffusion_steps", 0)),
        "samples": len(samples),
        "mean_learned_center_error": float(np.mean(learned_errors)),
        "mean_constant_velocity_center_error": float(np.mean(cv_errors)),
        "median_learned_center_error": float(np.median(learned_errors)),
        "median_constant_velocity_center_error": float(np.median(cv_errors)),
    }
    return ActionPolicyEvalResult(out_path=out_path, summary=summary)


def run_action_policy_ablation(
    jsonl_path: str | Path,
    out_dir: str | Path,
    model_types: list[str] | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 64,
    balance_by: list[str] | None = None,
    diffusion_steps: int = 16,
) -> ActionPolicyEvalResult:
    models = model_types or ["mlp", "residual_mlp", "diffusion"]
    if not models:
        raise ValueError("model_types must contain at least one backend")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    model_summaries: dict[str, Any] = {}
    cv_reference = None
    for model_type in models:
        safe_name = model_type.replace("/", "_")
        weights = out_root / f"action_policy_{safe_name}.pt"
        scores = out_root / f"action_policy_{safe_name}_sample_scores.jsonl"
        train_action_chunk_policy(
            jsonl_path,
            weights,
            epochs=epochs,
            lr=lr,
            hidden=hidden,
            batch_size=batch_size,
            balance_by=balance_by,
            model_type=model_type,
            diffusion_steps=diffusion_steps,
        )
        eval_result = evaluate_action_chunk_policy(jsonl_path, weights, scores)
        summary = dict(eval_result.summary)
        learned = float(summary["mean_learned_center_error"])
        cv = float(summary["mean_constant_velocity_center_error"])
        cv_reference = cv if cv_reference is None else cv_reference
        row = {
            "model_type": model_type,
            "weights": str(weights),
            "scores": str(scores),
            "samples": int(summary["samples"]),
            "mean_learned_center_error": learned,
            "mean_constant_velocity_center_error": cv,
            "mean_error_improvement_vs_cv": cv - learned,
            "median_learned_center_error": float(summary["median_learned_center_error"]),
            "median_constant_velocity_center_error": float(summary["median_constant_velocity_center_error"]),
            "diffusion_steps": int(summary.get("diffusion_steps", 0)),
        }
        rows.append(row)
        model_summaries[model_type] = {**summary, "weights": str(weights), "scores": str(scores)}

    best = min(rows, key=lambda row: row["mean_learned_center_error"])
    csv_path = out_root / "action_policy_ablation.csv"
    json_path = out_root / "action_policy_ablation_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "samples",
                "mean_learned_center_error",
                "mean_constant_velocity_center_error",
                "mean_error_improvement_vs_cv",
                "median_learned_center_error",
                "median_constant_velocity_center_error",
                "diffusion_steps",
                "weights",
                "scores",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "jsonl": str(jsonl_path),
        "out_dir": str(out_root),
        "csv": str(csv_path),
        "json": str(json_path),
        "model_types": models,
        "epochs": epochs,
        "lr": lr,
        "hidden": hidden,
        "batch_size": batch_size,
        "balance_by": balance_by,
        "diffusion_steps": diffusion_steps,
        "constant_velocity_reference_mean_error": float(cv_reference if cv_reference is not None else 0.0),
        "best": best,
        "models": model_summaries,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=csv_path, summary=summary)


def run_action_policy_split_selection(
    train_jsonl: str | Path,
    calib_jsonl: str | Path,
    out_dir: str | Path,
    test_jsonl: str | Path | None = None,
    model_types: list[str] | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 64,
    balance_by: list[str] | None = None,
    diffusion_steps: int = 16,
) -> ActionPolicyEvalResult:
    models = model_types or ["mlp", "residual_mlp", "diffusion"]
    if not models:
        raise ValueError("model_types must contain at least one backend")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    for model_type in models:
        safe_name = model_type.replace("/", "_")
        weights = out_root / f"action_policy_{safe_name}.pt"
        calib_scores = out_root / f"calib_scores_{safe_name}.jsonl"
        test_scores = out_root / f"test_scores_{safe_name}.jsonl"
        train_action_chunk_policy(
            train_jsonl,
            weights,
            epochs=epochs,
            lr=lr,
            hidden=hidden,
            batch_size=batch_size,
            balance_by=balance_by,
            model_type=model_type,
            diffusion_steps=diffusion_steps,
        )
        calib_eval = evaluate_action_chunk_policy(calib_jsonl, weights, calib_scores)
        test_eval = evaluate_action_chunk_policy(test_jsonl, weights, test_scores) if test_jsonl is not None else None
        calib_summary = calib_eval.summary
        test_summary = test_eval.summary if test_eval is not None else None
        row = {
            "model_type": model_type,
            "weights": str(weights),
            "calib_scores": str(calib_scores),
            "test_scores": str(test_scores) if test_eval is not None else "",
            "calib_samples": int(calib_summary.get("samples", 0)),
            "calib_mean_learned_center_error": float(calib_summary.get("mean_learned_center_error", 0.0)),
            "calib_mean_constant_velocity_center_error": float(calib_summary.get("mean_constant_velocity_center_error", 0.0)),
            "calib_mean_error_improvement_vs_cv": float(calib_summary.get("mean_constant_velocity_center_error", 0.0))
            - float(calib_summary.get("mean_learned_center_error", 0.0)),
            "test_samples": int(test_summary.get("samples", 0)) if test_summary is not None else 0,
            "test_mean_learned_center_error": float(test_summary.get("mean_learned_center_error", 0.0)) if test_summary is not None else 0.0,
            "test_mean_constant_velocity_center_error": float(test_summary.get("mean_constant_velocity_center_error", 0.0)) if test_summary is not None else 0.0,
            "test_mean_error_improvement_vs_cv": (
                float(test_summary.get("mean_constant_velocity_center_error", 0.0))
                - float(test_summary.get("mean_learned_center_error", 0.0))
                if test_summary is not None
                else 0.0
            ),
            "diffusion_steps": diffusion_steps if model_type == "diffusion" else 0,
        }
        rows.append(row)
        model_summaries[model_type] = {"calib": calib_summary, "test": test_summary}

    best = min(rows, key=lambda row: row["calib_mean_learned_center_error"])
    csv_path = out_root / "action_policy_split_selection.csv"
    json_path = out_root / "action_policy_split_selection_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "calib_samples",
                "calib_mean_learned_center_error",
                "calib_mean_constant_velocity_center_error",
                "calib_mean_error_improvement_vs_cv",
                "test_samples",
                "test_mean_learned_center_error",
                "test_mean_constant_velocity_center_error",
                "test_mean_error_improvement_vs_cv",
                "diffusion_steps",
                "weights",
                "calib_scores",
                "test_scores",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "train_jsonl": str(train_jsonl),
        "calib_jsonl": str(calib_jsonl),
        "test_jsonl": str(test_jsonl) if test_jsonl is not None else None,
        "out_dir": str(out_root),
        "csv": str(csv_path),
        "json": str(json_path),
        "model_types": models,
        "epochs": epochs,
        "lr": lr,
        "hidden": hidden,
        "batch_size": batch_size,
        "balance_by": balance_by,
        "diffusion_steps": diffusion_steps,
        "selection_metric": "calib_mean_learned_center_error",
        "best": best,
        "models": model_summaries,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=csv_path, summary=summary)


def run_multisource_action_policy_experiment(
    inputs: list[str | Path],
    out_dir: str | Path,
    source_names: list[str] | None = None,
    calib_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 59,
    group_field: str = "seq",
    source_field: str = "dataset_source",
    model_types: list[str] | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 64,
    balance_by: list[str] | None = None,
    diffusion_steps: int = 16,
) -> ActionPolicyEvalResult:
    """Run the Octo-style multi-source Route B action-policy experiment.

    The wrapper keeps the reproducibility contract in one directory: merge
    per-dataset action chunks, split by source/group to avoid clip leakage,
    then select the best dynamics backend on the calibration split.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    merged_jsonl = out_root / "merged_action_chunks.jsonl"
    merged_manifest = out_root / "merged_action_chunks.manifest.json"
    merge_result = merge_action_chunk_datasets(
        inputs,
        merged_jsonl,
        source_names=source_names,
        manifest_out=merged_manifest,
    )

    split_dir = out_root / "split"
    split_result = split_action_chunk_dataset(
        merged_jsonl,
        split_dir,
        calib_fraction=calib_fraction,
        test_fraction=test_fraction,
        seed=seed,
        group_field=group_field,
        source_field=source_field,
    )
    split_summary = split_result.summary
    if int(split_summary.get("train_samples", 0)) <= 0:
        raise ValueError("multi-source experiment produced an empty train split")
    if int(split_summary.get("calib_samples", 0)) <= 0:
        raise ValueError("multi-source experiment produced an empty calibration split")
    test_jsonl = split_summary.get("test_jsonl") if int(split_summary.get("test_samples", 0)) > 0 else None

    selection_dir = out_root / "selection"
    selection_result = run_action_policy_split_selection(
        split_summary["train_jsonl"],
        split_summary["calib_jsonl"],
        selection_dir,
        test_jsonl=test_jsonl,
        model_types=model_types,
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        batch_size=batch_size,
        balance_by=balance_by,
        diffusion_steps=diffusion_steps,
    )

    summary_path = out_root / "multisource_action_policy_experiment_summary.json"
    summary = {
        "out_dir": str(out_root),
        "inputs": [str(path) for path in inputs],
        "source_names": source_names,
        "merged_jsonl": str(merged_jsonl),
        "merged_manifest": str(merged_manifest),
        "split_dir": str(split_dir),
        "selection_dir": str(selection_dir),
        "summary_json": str(summary_path),
        "calib_fraction": calib_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "group_field": group_field,
        "source_field": source_field,
        "model_types": model_types or ["mlp", "residual_mlp", "diffusion"],
        "epochs": epochs,
        "lr": lr,
        "hidden": hidden,
        "batch_size": batch_size,
        "balance_by": balance_by,
        "diffusion_steps": diffusion_steps,
        "merge": merge_result.summary,
        "split": split_summary,
        "selection": selection_result.summary,
        "best": selection_result.summary.get("best"),
        "leakage_guard": split_summary.get("leakage_guard"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=selection_result.out_path, summary=summary)


def run_multisource_tracklet_action_policy_experiment(
    tracklet_inputs: list[str | Path],
    out_dir: str | Path,
    source_names: list[str] | None = None,
    past_len: int = 8,
    future_len: int = 8,
    image_size: tuple[int, int] | None = None,
    positives_only: bool = False,
    min_tracklet_rows: int = 0,
    calib_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 59,
    group_field: str = "seq",
    source_field: str = "dataset_source",
    model_types: list[str] | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 64,
    balance_by: list[str] | None = None,
    diffusion_steps: int = 16,
) -> ActionPolicyEvalResult:
    if not tracklet_inputs:
        raise ValueError("tracklet_inputs must contain at least one tracklet JSONL")
    if source_names is not None and len(source_names) != len(tracklet_inputs):
        raise ValueError("source_names must have the same length as tracklet_inputs")

    out_root = Path(out_dir)
    action_chunk_dir = out_root / "action_chunks"
    action_chunk_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: list[Path] = []
    export_summaries: list[dict[str, Any]] = []
    for index, tracklet_input in enumerate(tracklet_inputs):
        source_name = source_names[index] if source_names is not None else Path(tracklet_input).stem
        safe_source = str(source_name).replace("/", "_").replace("\\", "_") or f"source_{index}"
        action_chunk_path = action_chunk_dir / f"{safe_source}_action_chunks.jsonl"
        export_result = export_action_chunk_dataset_from_tracklets(
            tracklet_input,
            action_chunk_path,
            past_len=past_len,
            future_len=future_len,
            image_size=image_size,
            positives_only=positives_only,
            min_tracklet_rows=min_tracklet_rows,
        )
        if int(export_result.summary.get("samples", 0)) <= 0:
            raise ValueError(f"tracklet source produced no action chunks: {tracklet_input}")
        exported_paths.append(export_result.jsonl_path)
        export_summaries.append(
            {
                "source_name": source_name,
                "tracklet_input": str(tracklet_input),
                **export_result.summary,
            }
        )

    policy_result = run_multisource_action_policy_experiment(
        exported_paths,
        out_root / "policy",
        source_names=source_names,
        calib_fraction=calib_fraction,
        test_fraction=test_fraction,
        seed=seed,
        group_field=group_field,
        source_field=source_field,
        model_types=model_types,
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        batch_size=batch_size,
        balance_by=balance_by,
        diffusion_steps=diffusion_steps,
    )

    summary_path = out_root / "multisource_tracklet_action_policy_experiment_summary.json"
    summary = {
        "out_dir": str(out_root),
        "tracklet_inputs": [str(path) for path in tracklet_inputs],
        "source_names": source_names,
        "action_chunk_dir": str(action_chunk_dir),
        "policy_dir": str(out_root / "policy"),
        "summary_json": str(summary_path),
        "past_len": past_len,
        "future_len": future_len,
        "image_size": list(image_size) if image_size is not None else None,
        "positives_only": positives_only,
        "min_tracklet_rows": min_tracklet_rows,
        "calib_fraction": calib_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "group_field": group_field,
        "source_field": source_field,
        "model_types": model_types or ["mlp", "residual_mlp", "diffusion"],
        "epochs": epochs,
        "lr": lr,
        "hidden": hidden,
        "batch_size": batch_size,
        "balance_by": balance_by,
        "diffusion_steps": diffusion_steps,
        "exports": export_summaries,
        "policy": policy_result.summary,
        "best": policy_result.summary.get("best"),
        "leakage_guard": policy_result.summary.get("leakage_guard"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=policy_result.out_path, summary=summary)


def run_multisource_tracklet_policy_benchmark(
    train_tracklet_inputs: list[str | Path],
    eval_tracklet_inputs: list[str | Path],
    out_dir: str | Path,
    train_source_names: list[str] | None = None,
    eval_dataset_names: list[str] | None = None,
    past_len: int = 8,
    future_len: int = 8,
    image_size: tuple[int, int] | None = None,
    positives_only: bool = False,
    min_tracklet_rows: int = 0,
    calib_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 59,
    group_field: str = "seq",
    source_field: str = "dataset_source",
    model_types: list[str] | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 64,
    balance_by: list[str] | None = None,
    diffusion_steps: int = 16,
    error_scale: float = 8.0,
    thresholds: list[float] | None = None,
    baseline_csv: str | Path | None = None,
    baseline_metric: str = "best_f1",
    baseline_lower_is_better: bool = False,
    baseline_digits: int = 3,
    allow_invalid_baselines: bool = False,
) -> ActionPolicyEvalResult:
    if not eval_tracklet_inputs:
        raise ValueError("eval_tracklet_inputs must contain at least one held-out tracklet JSONL")
    if eval_dataset_names is not None and len(eval_dataset_names) != len(eval_tracklet_inputs):
        raise ValueError("eval_dataset_names must have the same length as eval_tracklet_inputs")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    train_result = run_multisource_tracklet_action_policy_experiment(
        train_tracklet_inputs,
        out_root / "train",
        source_names=train_source_names,
        past_len=past_len,
        future_len=future_len,
        image_size=image_size,
        positives_only=positives_only,
        min_tracklet_rows=min_tracklet_rows,
        calib_fraction=calib_fraction,
        test_fraction=test_fraction,
        seed=seed,
        group_field=group_field,
        source_field=source_field,
        model_types=model_types,
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        batch_size=batch_size,
        balance_by=balance_by,
        diffusion_steps=diffusion_steps,
    )
    best = dict(train_result.summary.get("best") or {})
    weights = best.get("weights")
    if not weights:
        raise ValueError("multi-source training did not produce selected weights")

    eval_summaries = []
    eval_summary_paths: list[str] = []
    dataset_names: list[str] = []
    eval_root = out_root / "eval"
    for index, eval_tracklet in enumerate(eval_tracklet_inputs):
        dataset_name = eval_dataset_names[index] if eval_dataset_names is not None else Path(eval_tracklet).stem
        safe_dataset = str(dataset_name).replace("/", "_").replace("\\", "_") or f"dataset_{index}"
        dataset_dir = eval_root / safe_dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        tracklet_scores = dataset_dir / "tracklet_dynamics_scores.jsonl"
        attached_tracklets = dataset_dir / "tracklets_with_action_dynamics.jsonl"

        tracklet_eval = score_tracklets_with_action_policy(
            eval_tracklet,
            weights,
            tracklet_scores,
            past_len=past_len,
            future_len=future_len,
            image_size=image_size,
            error_scale=error_scale,
            min_tracklet_rows=min_tracklet_rows,
        )
        attach_eval = attach_action_dynamics_scores_to_tracklets(eval_tracklet, tracklet_scores, attached_tracklets)
        threshold_eval = evaluate_action_dynamics_thresholds(tracklet_scores, dataset_dir, thresholds=thresholds)
        summary_path = dataset_dir / "action_dynamics_pipeline_summary.json"
        eval_summary = {
            "tracklet_jsonl": str(eval_tracklet),
            "out_dir": str(dataset_dir),
            "past_len": past_len,
            "future_len": future_len,
            "image_size": list(image_size) if image_size is not None else None,
            "min_tracklet_rows": min_tracklet_rows,
            "error_scale": error_scale,
            "thresholds": thresholds,
            "model_type": str(best.get("model_type", train_result.summary.get("policy", {}).get("best", {}).get("model_type", ""))),
            "diffusion_steps": diffusion_steps if str(best.get("model_type", "")) == "diffusion" else 0,
            "policy_weights": str(weights),
            "tracklet_scores": str(tracklet_scores),
            "attached_tracklets": str(attached_tracklets),
            "final_tracklets": str(attached_tracklets),
            "threshold_sweep": str(threshold_eval.out_path),
            "tracklet_eval": tracklet_eval.summary,
            "attach": attach_eval.summary,
            "threshold_eval": threshold_eval.summary,
            "training_summary": train_result.summary.get("summary_json"),
        }
        summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        eval_summaries.append(eval_summary)
        eval_summary_paths.append(str(summary_path))
        dataset_names.append(str(dataset_name))

    collected = collect_route_b_result_summaries(
        eval_summary_paths,
        out_root / "collected",
        dataset_names=dataset_names,
    )
    baseline_report = None
    if baseline_csv is not None:
        baseline_report = build_route_b_baseline_report(
            eval_summary_paths,
            baseline_csv,
            out_root / "baseline_report",
            dataset_names=dataset_names,
            metric=baseline_metric,
            higher_is_better=not baseline_lower_is_better,
            digits=baseline_digits,
            strict_baselines=not allow_invalid_baselines,
        )
    summary_path = out_root / "multisource_tracklet_policy_benchmark_summary.json"
    summary = {
        "out_dir": str(out_root),
        "summary_json": str(summary_path),
        "train": train_result.summary,
        "selected_policy": best,
        "selected_weights": str(weights),
        "eval_tracklet_inputs": [str(path) for path in eval_tracklet_inputs],
        "eval_dataset_names": dataset_names,
        "eval_summaries": eval_summaries,
        "eval_summary_paths": eval_summary_paths,
        "collected": collected.summary,
        "results_csv": str(collected.out_path),
        "baseline_csv": str(baseline_csv) if baseline_csv is not None else None,
        "baseline_metric": baseline_metric,
        "baseline_lower_is_better": baseline_lower_is_better,
        "baseline_report": baseline_report.summary if baseline_report is not None else None,
        "baseline_report_out": str(baseline_report.out_path) if baseline_report is not None else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=collected.out_path, summary=summary)


def run_multisource_proposal_policy_benchmark(
    train_run_roots: list[str | Path],
    train_gt_csvs: list[str | Path],
    eval_run_roots: list[str | Path],
    eval_gt_csvs: list[str | Path],
    out_dir: str | Path,
    train_source_names: list[str] | None = None,
    eval_dataset_names: list[str] | None = None,
    profile: str = "hard_recovery",
    diagnostics_name: str = "diagnostics_raw.jsonl",
    max_frames: int | None = None,
    proposal_max_gap: int = 3,
    proposal_base_radius: float = 18.0,
    proposal_radius_per_side: float = 0.75,
    proposal_min_iou: float = 0.05,
    proposal_min_score: float = 0.0,
    proposal_detector_only: bool = False,
    proposal_min_tracklet_rows: int = 1,
    proposal_iou_threshold: float = 0.3,
    proposal_center_threshold: float = 24.0,
    proposal_hard_tiny_side: float = 24.0,
    proposal_hard_low_score: float = 0.25,
    past_len: int = 8,
    future_len: int = 8,
    image_size: tuple[int, int] | None = None,
    positives_only: bool = False,
    min_tracklet_rows: int = 0,
    calib_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 59,
    group_field: str = "seq",
    source_field: str = "dataset_source",
    model_types: list[str] | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 64,
    balance_by: list[str] | None = None,
    diffusion_steps: int = 16,
    error_scale: float = 8.0,
    thresholds: list[float] | None = None,
    baseline_csv: str | Path | None = None,
    baseline_metric: str = "best_f1",
    baseline_lower_is_better: bool = False,
    baseline_digits: int = 3,
    allow_invalid_baselines: bool = False,
) -> ActionPolicyEvalResult:
    if not train_run_roots:
        raise ValueError("train_run_roots must contain at least one run root")
    if not eval_run_roots:
        raise ValueError("eval_run_roots must contain at least one run root")
    if len(train_run_roots) != len(train_gt_csvs):
        raise ValueError("train_run_roots and train_gt_csvs must have the same length")
    if len(eval_run_roots) != len(eval_gt_csvs):
        raise ValueError("eval_run_roots and eval_gt_csvs must have the same length")
    if train_source_names is not None and len(train_source_names) != len(train_run_roots):
        raise ValueError("train_source_names must have the same length as train_run_roots")
    if eval_dataset_names is not None and len(eval_dataset_names) != len(eval_run_roots):
        raise ValueError("eval_dataset_names must have the same length as eval_run_roots")

    out_root = Path(out_dir)
    proposal_root = out_root / "proposal_tracklets"
    train_tracklets: list[Path] = []
    eval_tracklets: list[Path] = []
    proposal_summaries: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}

    def build_one(split: str, index: int, run_root: str | Path, gt_csv: str | Path, name: str) -> Path:
        safe_name = str(name).replace("/", "_").replace("\\", "_") or f"{split}_{index}"
        result = build_proposal_tracklet_dataset(
            [run_root],
            gt_csv,
            proposal_root / split / safe_name,
            profile=profile,
            diagnostics_name=diagnostics_name,
            max_frames=max_frames,
            max_gap=proposal_max_gap,
            base_radius=proposal_base_radius,
            radius_per_side=proposal_radius_per_side,
            min_iou=proposal_min_iou,
            min_score=proposal_min_score,
            detector_only=proposal_detector_only,
            min_tracklet_rows=proposal_min_tracklet_rows,
            iou_threshold=proposal_iou_threshold,
            center_threshold=proposal_center_threshold,
            hard_tiny_side=proposal_hard_tiny_side,
            hard_low_score=proposal_hard_low_score,
        )
        proposal_summaries[split].append(
            {
                "name": name,
                "run_root": str(run_root),
                "gt_csv": str(gt_csv),
                "tracklet_jsonl": str(result.json_path),
                "tracklet_csv": str(result.csv_path),
                **result.summary,
            }
        )
        if int(result.summary.get("num_tracklets", 0)) <= 0:
            raise ValueError(f"{split}/{name} produced zero proposal tracklets")
        return result.json_path

    for index, (run_root, gt_csv) in enumerate(zip(train_run_roots, train_gt_csvs)):
        name = train_source_names[index] if train_source_names is not None else Path(run_root).name
        train_tracklets.append(build_one("train", index, run_root, gt_csv, str(name)))
    for index, (run_root, gt_csv) in enumerate(zip(eval_run_roots, eval_gt_csvs)):
        name = eval_dataset_names[index] if eval_dataset_names is not None else Path(run_root).name
        eval_tracklets.append(build_one("eval", index, run_root, gt_csv, str(name)))

    benchmark = run_multisource_tracklet_policy_benchmark(
        train_tracklets,
        eval_tracklets,
        out_root / "benchmark",
        train_source_names=train_source_names,
        eval_dataset_names=eval_dataset_names,
        past_len=past_len,
        future_len=future_len,
        image_size=image_size,
        positives_only=positives_only,
        min_tracklet_rows=min_tracklet_rows,
        calib_fraction=calib_fraction,
        test_fraction=test_fraction,
        seed=seed,
        group_field=group_field,
        source_field=source_field,
        model_types=model_types,
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        batch_size=batch_size,
        balance_by=balance_by,
        diffusion_steps=diffusion_steps,
        error_scale=error_scale,
        thresholds=thresholds,
        baseline_csv=baseline_csv,
        baseline_metric=baseline_metric,
        baseline_lower_is_better=baseline_lower_is_better,
        baseline_digits=baseline_digits,
        allow_invalid_baselines=allow_invalid_baselines,
    )

    summary_path = out_root / "multisource_proposal_policy_benchmark_summary.json"
    summary = {
        "out_dir": str(out_root),
        "summary_json": str(summary_path),
        "profile": profile,
        "diagnostics_name": diagnostics_name,
        "max_frames": max_frames,
        "proposal_params": {
            "max_gap": proposal_max_gap,
            "base_radius": proposal_base_radius,
            "radius_per_side": proposal_radius_per_side,
            "min_iou": proposal_min_iou,
            "min_score": proposal_min_score,
            "detector_only": proposal_detector_only,
            "min_tracklet_rows": proposal_min_tracklet_rows,
            "iou_threshold": proposal_iou_threshold,
            "center_threshold": proposal_center_threshold,
            "hard_tiny_side": proposal_hard_tiny_side,
            "hard_low_score": proposal_hard_low_score,
        },
        "proposal_tracklets": proposal_summaries,
        "train_tracklet_inputs": [str(path) for path in train_tracklets],
        "eval_tracklet_inputs": [str(path) for path in eval_tracklets],
        "benchmark": benchmark.summary,
        "results_csv": str(benchmark.out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=benchmark.out_path, summary=summary)


def _load_tracklet_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def _inspect_tracklet_input(
    path: str | Path,
    name: str,
    past_len: int,
    future_len: int,
    min_tracklet_rows: int,
) -> dict[str, Any]:
    path_obj = Path(path)
    summary: dict[str, Any] = {
        "name": name,
        "path": str(path_obj),
        "exists": path_obj.exists(),
        "tracklets": 0,
        "usable_tracklets": 0,
        "rows": 0,
        "action_chunk_samples": 0,
        "positive_tracklets": 0,
        "negative_tracklets": 0,
        "unlabeled_tracklets": 0,
        "min_rows": 0,
        "max_rows": 0,
        "issues": [],
        "warnings": [],
    }
    if not path_obj.exists():
        summary["issues"].append("file does not exist")
        return summary

    try:
        items = _load_tracklet_jsonl(path_obj)
    except Exception as exc:
        summary["issues"].append(f"failed to parse JSONL: {exc}")
        return summary

    row_counts = []
    total_required = past_len + future_len
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            summary["issues"].append(f"line {index + 1}: item is not an object")
            continue
        meta = item.get("meta") or {}
        rows = item.get("rows") or []
        if not isinstance(rows, list):
            summary["issues"].append(f"line {index + 1}: rows is not a list")
            continue
        summary["tracklets"] += 1
        summary["rows"] += len(rows)
        row_counts.append(len(rows))
        label_raw = meta.get("label", None) if isinstance(meta, dict) else None
        if label_raw is None:
            summary["unlabeled_tracklets"] += 1
        else:
            label = int(float(label_raw))
            if label > 0:
                summary["positive_tracklets"] += 1
            else:
                summary["negative_tracklets"] += 1
        if min_tracklet_rows > 0 and len(rows) < min_tracklet_rows:
            continue
        try:
            samples = build_action_chunk_samples_from_rows(
                rows,
                past_len=past_len,
                future_len=future_len,
                seq=str(meta.get("seq", "")) if isinstance(meta, dict) else "",
                track_id=str(meta.get("track_id", "")) if isinstance(meta, dict) else "",
            )
        except Exception as exc:
            summary["issues"].append(f"line {index + 1}: cannot build action chunks: {exc}")
            continue
        if len(rows) >= total_required and not samples:
            summary["warnings"].append(f"line {index + 1}: has enough rows but produced no samples")
        if samples:
            summary["usable_tracklets"] += 1
            summary["action_chunk_samples"] += len(samples)

    if row_counts:
        summary["min_rows"] = int(min(row_counts))
        summary["max_rows"] = int(max(row_counts))
    if summary["tracklets"] == 0:
        summary["issues"].append("no tracklets found")
    if summary["action_chunk_samples"] == 0:
        summary["issues"].append(f"no action chunks possible with past_len={past_len}, future_len={future_len}")
    if summary["positive_tracklets"] == 0:
        summary["warnings"].append("no positive tracklets")
    if summary["negative_tracklets"] == 0:
        summary["warnings"].append("no negative tracklets")
    return summary


def validate_route_b_tracklet_inputs(
    train_tracklet_inputs: list[str | Path],
    eval_tracklet_inputs: list[str | Path],
    out: str | Path,
    train_source_names: list[str] | None = None,
    eval_dataset_names: list[str] | None = None,
    past_len: int = 8,
    future_len: int = 8,
    min_tracklet_rows: int = 0,
) -> ActionPolicyEvalResult:
    issues: list[str] = []
    warnings: list[str] = []
    if past_len <= 0 or future_len <= 0:
        issues.append("past_len and future_len must be positive")
    if min_tracklet_rows < 0:
        issues.append("min_tracklet_rows must be nonnegative")
    if not train_tracklet_inputs:
        issues.append("at least one train tracklet input is required")
    if not eval_tracklet_inputs:
        issues.append("at least one eval tracklet input is required")
    if train_source_names is not None and len(train_source_names) != len(train_tracklet_inputs):
        issues.append("train_source_names must have the same length as train_tracklet_inputs")
    if eval_dataset_names is not None and len(eval_dataset_names) != len(eval_tracklet_inputs):
        issues.append("eval_dataset_names must have the same length as eval_tracklet_inputs")

    train_summaries = []
    eval_summaries = []
    if past_len > 0 and future_len > 0 and min_tracklet_rows >= 0:
        for index, path in enumerate(train_tracklet_inputs):
            name = train_source_names[index] if train_source_names is not None else Path(path).stem
            summary = _inspect_tracklet_input(path, str(name), past_len, future_len, min_tracklet_rows)
            train_summaries.append(summary)
            issues.extend(f"train/{name}: {issue}" for issue in summary["issues"])
            warnings.extend(f"train/{name}: {warning}" for warning in summary["warnings"])
        for index, path in enumerate(eval_tracklet_inputs):
            name = eval_dataset_names[index] if eval_dataset_names is not None else Path(path).stem
            summary = _inspect_tracklet_input(path, str(name), past_len, future_len, min_tracklet_rows)
            eval_summaries.append(summary)
            issues.extend(f"eval/{name}: {issue}" for issue in summary["issues"])
            warnings.extend(f"eval/{name}: {warning}" for warning in summary["warnings"])
            if summary.get("unlabeled_tracklets", 0) > 0:
                issues.append(f"eval/{name}: unlabeled tracklets cannot be threshold-evaluated")

    train_samples = sum(int(s.get("action_chunk_samples", 0)) for s in train_summaries)
    eval_samples = sum(int(s.get("action_chunk_samples", 0)) for s in eval_summaries)
    if train_summaries and train_samples == 0:
        issues.append("all train inputs have zero action-chunk samples")
    if eval_summaries and eval_samples == 0:
        issues.append("all eval inputs have zero action-chunk samples")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "valid": len(issues) == 0,
        "out": str(out_path),
        "past_len": past_len,
        "future_len": future_len,
        "min_tracklet_rows": min_tracklet_rows,
        "train_tracklet_inputs": [str(path) for path in train_tracklet_inputs],
        "eval_tracklet_inputs": [str(path) for path in eval_tracklet_inputs],
        "train_source_names": train_source_names,
        "eval_dataset_names": eval_dataset_names,
        "train": train_summaries,
        "eval": eval_summaries,
        "train_action_chunk_samples": train_samples,
        "eval_action_chunk_samples": eval_samples,
        "issues": issues,
        "warnings": warnings,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=out_path, summary=summary)


def score_tracklets_with_action_policy(
    tracklet_jsonl: str | Path,
    weights: str | Path,
    out: str | Path,
    past_len: int = 8,
    future_len: int = 8,
    image_size: tuple[int, int] | None = None,
    normalize_by_row_image_size: bool = False,
    error_scale: float = 8.0,
    dynamics_score_mode: str = "learned_consistency",
    min_tracklet_rows: int = 0,
) -> ActionPolicyEvalResult:
    if dynamics_score_mode not in {"learned_consistency", "cv_consistency", "improvement", "hybrid"}:
        raise ValueError("dynamics_score_mode must be learned_consistency, cv_consistency, improvement, or hybrid")
    items = _load_tracklet_jsonl(tracklet_jsonl)
    model, ckpt = _load_policy(weights)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    learned_all = []
    cv_all = []
    scored_tracklets = 0
    unscored_tracklets = 0
    with torch.no_grad():
        for item in items:
            meta = dict(item.get("meta") or {})
            rows = list(item.get("rows") or [])
            if min_tracklet_rows > 0 and len(rows) < min_tracklet_rows:
                unscored_tracklets += 1
                continue
            samples = build_action_chunk_samples_from_rows(
                rows,
                past_len=past_len,
                future_len=future_len,
                image_size=image_size,
                normalize_by_row_image_size=normalize_by_row_image_size,
                seq=str(meta.get("seq", "")),
                track_id=str(meta.get("track_id", "")),
            )
            if not samples:
                unscored_tracklets += 1
                continue
            learned_errors = []
            cv_errors = []
            sample_rows = []
            for sample in samples:
                sample_dict = {
                    "seq": sample.seq,
                    "track_id": sample.track_id,
                    "anchor_frame": sample.anchor_frame,
                    "past_boxes": sample.past_boxes.tolist(),
                    "past_scores": sample.past_scores.tolist(),
                    "past_visible": sample.past_visible.tolist(),
                    "future_actions": sample.future_actions.tolist(),
                    "future_boxes": sample.future_boxes.tolist(),
                }
                learned_error, cv_error, _, _, _, _ = _predict_sample_errors(sample_dict, model, ckpt)
                learned_errors.append(learned_error)
                cv_errors.append(cv_error)
                sample_rows.append(
                    {
                        "anchor_frame": sample.anchor_frame,
                        "learned_center_error": learned_error,
                        "constant_velocity_center_error": cv_error,
                    }
                )
            mean_learned = float(np.mean(learned_errors))
            mean_cv = float(np.mean(cv_errors))
            scale = max(float(error_scale), 1e-6)
            learned_consistency = float(np.exp(-mean_learned / scale))
            cv_consistency = float(np.exp(-mean_cv / scale))
            improvement_score = _stable_sigmoid((mean_cv - mean_learned) / scale)
            if dynamics_score_mode == "cv_consistency":
                dynamics_score = cv_consistency
            elif dynamics_score_mode == "improvement":
                dynamics_score = improvement_score
            elif dynamics_score_mode == "hybrid":
                dynamics_score = learned_consistency * improvement_score
            else:
                dynamics_score = learned_consistency
            row_out = {
                "seq": str(meta.get("seq", "")),
                "track_id": str(meta.get("track_id", "")),
                "label": int(float(meta.get("label", 0))),
                "bucket": str(meta.get("bucket", "")),
                "dataset_source": str(meta.get("dataset_source", "")),
                "num_rows": len(rows),
                "num_action_windows": len(samples),
                "mean_learned_center_error": mean_learned,
                "mean_constant_velocity_center_error": mean_cv,
                "median_learned_center_error": float(np.median(learned_errors)),
                "median_constant_velocity_center_error": float(np.median(cv_errors)),
                "mean_error_improvement_vs_cv": mean_cv - mean_learned,
                "learned_consistency_score": learned_consistency,
                "constant_velocity_consistency_score": cv_consistency,
                "improvement_score": improvement_score,
                "dynamics_score_mode": dynamics_score_mode,
                "dynamics_score": dynamics_score,
                "sample_scores": sample_rows,
            }
            rows_out.append(row_out)
            learned_all.extend(learned_errors)
            cv_all.extend(cv_errors)
            scored_tracklets += 1
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "weights": str(weights),
        "out": str(out_path),
        "model_type": str(ckpt.get("model_type", "mlp")),
        "diffusion_steps": int(ckpt.get("diffusion_steps", 0)),
        "past_len": past_len,
        "future_len": future_len,
        "image_size": list(image_size) if image_size is not None else None,
        "normalize_by_row_image_size": normalize_by_row_image_size,
        "error_scale": error_scale,
        "dynamics_score_mode": dynamics_score_mode,
        "total_tracklets": len(items),
        "scored_tracklets": scored_tracklets,
        "unscored_tracklets": unscored_tracklets,
        "action_windows": len(learned_all),
        "mean_learned_center_error": float(np.mean(learned_all)) if learned_all else 0.0,
        "mean_constant_velocity_center_error": float(np.mean(cv_all)) if cv_all else 0.0,
        "mean_error_improvement_vs_cv": float(np.mean(cv_all) - np.mean(learned_all)) if learned_all else 0.0,
    }
    return ActionPolicyEvalResult(out_path=out_path, summary=summary)


def attach_action_dynamics_scores_to_tracklets(
    tracklet_jsonl: str | Path,
    dynamics_scores_jsonl: str | Path,
    out: str | Path,
) -> ActionPolicyEvalResult:
    scores: dict[tuple[str, str], dict[str, Any]] = {}
    with Path(dynamics_scores_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scores[(str(row.get("seq", "")), str(row.get("track_id", "")))] = row

    items = _load_tracklet_jsonl(tracklet_jsonl)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    attached = 0
    missing = 0
    score_values = []
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            total += 1
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in (item.get("rows") or [])]
            key = (str(meta.get("seq", "")), str(meta.get("track_id", "")))
            score = scores.get(key)
            if score is None:
                missing += 1
            else:
                attached += 1
                dynamics_score = float(score.get("dynamics_score", 0.0))
                learned_error = float(score.get("mean_learned_center_error", 0.0))
                cv_error = float(score.get("mean_constant_velocity_center_error", 0.0))
                improvement = float(score.get("mean_error_improvement_vs_cv", cv_error - learned_error))
                meta.update(
                    {
                        "action_dynamics_score": dynamics_score,
                        "action_mean_learned_center_error": learned_error,
                        "action_mean_constant_velocity_center_error": cv_error,
                        "action_error_improvement_vs_cv": improvement,
                        "action_num_windows": int(score.get("num_action_windows", 0)),
                    }
                )
                extra_score_fields = (
                    "predicted_confidence_score",
                    "target_confidence_score",
                    "video_action_model_fusion_score",
                    "video_action_model_fusion_mode",
                    "mean_video_action_center_error",
                    "median_video_action_center_error",
                    "dynamics_score_mode",
                )
                for field in extra_score_fields:
                    if field in score:
                        meta[field] = score[field]
                for row in rows:
                    row["action_dynamics_score"] = dynamics_score
                    row["action_mean_learned_center_error"] = learned_error
                    row["action_mean_constant_velocity_center_error"] = cv_error
                    row["action_error_improvement_vs_cv"] = improvement
                    row["action_num_windows"] = int(score.get("num_action_windows", 0))
                    for field in extra_score_fields:
                        if field in score:
                            row[field] = score[field]
                score_values.append(dynamics_score)
            item["meta"] = meta
            item["rows"] = rows
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "dynamics_scores_jsonl": str(dynamics_scores_jsonl),
        "out": str(out_path),
        "total_tracklets": total,
        "attached_tracklets": attached,
        "missing_tracklets": missing,
        "mean_action_dynamics_score": float(np.mean(score_values)) if score_values else 0.0,
    }
    return ActionPolicyEvalResult(out_path=out_path, summary=summary)


def attach_tracklet_confidence_fusion_scores(
    tracklet_jsonl: str | Path,
    out: str | Path,
    action_score_field: str = "action_dynamics_score",
    confidence_fields: tuple[str, ...] = ("final_drone_score", "objectness", "score"),
    confidence_reduction: str = "mean",
    out_score_field: str = "video_action_conf_score",
    missing_action_score: float | None = None,
) -> ActionPolicyEvalResult:
    """Attach a reproducible tracklet-level fusion score to every scored tracklet."""
    if confidence_reduction not in {"mean", "max"}:
        raise ValueError("confidence_reduction must be 'mean' or 'max'")

    def row_confidence(row: dict[str, Any]) -> float:
        for field in confidence_fields:
            if field in row and row[field] is not None:
                return float(row[field])
        return 0.0

    items = _load_tracklet_jsonl(tracklet_jsonl)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    fused = 0
    missing = 0
    score_values: list[float] = []
    conf_values: list[float] = []
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            total += 1
            meta = dict(item.get("meta") or {})
            rows = [dict(row) for row in (item.get("rows") or [])]
            confidences = [row_confidence(row) for row in rows]
            if confidence_reduction == "max":
                tracklet_confidence = max(confidences, default=0.0)
            else:
                tracklet_confidence = float(np.mean(confidences)) if confidences else 0.0
            meta["tracklet_detection_confidence"] = tracklet_confidence
            meta["tracklet_detection_confidence_reduction"] = confidence_reduction

            raw_action_score = meta.get(action_score_field)
            if raw_action_score is None and missing_action_score is not None:
                raw_action_score = missing_action_score
            if raw_action_score is None:
                missing += 1
            else:
                action_score = float(raw_action_score)
                fused_score = action_score * tracklet_confidence
                fused += 1
                score_values.append(fused_score)
                conf_values.append(tracklet_confidence)
                meta[out_score_field] = fused_score
                meta[f"{out_score_field}_action_score"] = action_score
                for row in rows:
                    row["tracklet_detection_confidence"] = tracklet_confidence
                    row["tracklet_detection_confidence_reduction"] = confidence_reduction
                    row[out_score_field] = fused_score
                    row[f"{out_score_field}_action_score"] = action_score
            item["meta"] = meta
            item["rows"] = rows
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "out": str(out_path),
        "action_score_field": action_score_field,
        "confidence_fields": list(confidence_fields),
        "confidence_reduction": confidence_reduction,
        "out_score_field": out_score_field,
        "total_tracklets": total,
        "fused_tracklets": fused,
        "missing_action_score_tracklets": missing,
        "mean_fusion_score": float(np.mean(score_values)) if score_values else 0.0,
        "min_fusion_score": float(np.min(score_values)) if score_values else 0.0,
        "max_fusion_score": float(np.max(score_values)) if score_values else 0.0,
        "mean_tracklet_detection_confidence": float(np.mean(conf_values)) if conf_values else 0.0,
    }
    return ActionPolicyEvalResult(out_path=out_path, summary=summary)


def _tracklet_row_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bbox" in row:
        values = row["bbox"]
    else:
        values = [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")]
    if values is None or len(values) != 4:
        raise ValueError("tracklet row must contain bbox or x1/y1/x2/y2")
    x1, y1, x2, y2 = [float(v) for v in values]
    return x1, y1, x2, y2


def score_tracklets_with_constant_velocity(
    tracklet_jsonl: str | Path,
    out: str | Path,
    min_tracklet_rows: int = 3,
    error_scale: float = 1.0,
    min_box_side: float = 1.0,
) -> ActionPolicyEvalResult:
    if min_tracklet_rows < 3:
        raise ValueError("min_tracklet_rows must be >= 3 for constant-velocity scoring")
    if error_scale <= 0:
        raise ValueError("error_scale must be positive")
    items = _load_tracklet_jsonl(tracklet_jsonl)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    scored = 0
    unscored = 0
    all_errors: list[float] = []
    all_scores: list[float] = []

    for item in items:
        meta = dict(item.get("meta") or {})
        rows = sorted([dict(row) for row in (item.get("rows") or [])], key=lambda row: int(float(row.get("frame_id", 0) or 0)))
        if len(rows) < min_tracklet_rows:
            unscored += 1
            continue
        centers = []
        sides = []
        frame_ids = []
        try:
            for row in rows:
                x1, y1, x2, y2 = _tracklet_row_box(row)
                centers.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                sides.append(max(float(x2 - x1), float(y2 - y1), float(min_box_side)))
                frame_ids.append(int(float(row.get("frame_id", 0) or 0)))
        except (TypeError, ValueError):
            unscored += 1
            continue
        centers_arr = np.asarray(centers, dtype=np.float32)
        sides_arr = np.asarray(sides, dtype=np.float32)
        frame_arr = np.asarray(frame_ids, dtype=np.float32)
        errors = []
        window_rows = []
        for index in range(2, len(rows)):
            prev_dt = max(float(frame_arr[index - 1] - frame_arr[index - 2]), 1.0)
            next_dt = max(float(frame_arr[index] - frame_arr[index - 1]), 1.0)
            velocity = (centers_arr[index - 1] - centers_arr[index - 2]) / prev_dt
            predicted = centers_arr[index - 1] + velocity * next_dt
            center_error = float(np.linalg.norm(predicted - centers_arr[index]))
            denom = max(float(np.mean(sides_arr[index - 2 : index + 1])), float(min_box_side))
            normalized_error = center_error / denom
            errors.append(normalized_error)
            window_rows.append(
                {
                    "frame_id": int(frame_arr[index]),
                    "center_error": center_error,
                    "normalized_center_error": normalized_error,
                    "predicted_center": predicted.tolist(),
                    "actual_center": centers_arr[index].tolist(),
                }
            )
        if not errors:
            unscored += 1
            continue
        mean_error = float(np.mean(errors))
        median_error = float(np.median(errors))
        max_error = float(np.max(errors))
        score = float(np.exp(-mean_error / float(error_scale)))
        row_out = {
            "seq": str(meta.get("seq", rows[0].get("seq", ""))),
            "track_id": str(meta.get("track_id", rows[0].get("track_id", ""))),
            "raw_track_id": str(meta.get("raw_track_id", rows[0].get("raw_track_id", rows[0].get("track_id", "")))),
            "label": int(float(meta.get("label", 0))),
            "bucket": str(meta.get("bucket", "")),
            "dataset_source": str(meta.get("dataset_source", "")),
            "num_rows": len(rows),
            "num_velocity_windows": len(errors),
            "mean_cv_normalized_center_error": mean_error,
            "median_cv_normalized_center_error": median_error,
            "max_cv_normalized_center_error": max_error,
            "dynamics_score_mode": "constant_velocity_normalized",
            "dynamics_score": score,
            "sample_scores": window_rows,
        }
        rows_out.append(row_out)
        all_errors.extend(errors)
        all_scores.append(score)
        scored += 1

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "out": str(out_path),
        "min_tracklet_rows": min_tracklet_rows,
        "error_scale": error_scale,
        "min_box_side": min_box_side,
        "total_tracklets": len(items),
        "scored_tracklets": scored,
        "unscored_tracklets": unscored,
        "velocity_windows": len(all_errors),
        "mean_cv_normalized_center_error": float(np.mean(all_errors)) if all_errors else 0.0,
        "median_cv_normalized_center_error": float(np.median(all_errors)) if all_errors else 0.0,
        "mean_dynamics_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "score_min": float(np.min(all_scores)) if all_scores else 0.0,
        "score_max": float(np.max(all_scores)) if all_scores else 0.0,
    }
    return ActionPolicyEvalResult(out_path=out_path, summary=summary)


def run_action_dynamics_tracklet_pipeline(
    tracklet_jsonl: str | Path,
    out_dir: str | Path,
    past_len: int = 8,
    future_len: int = 8,
    image_size: tuple[int, int] | None = None,
    normalize_by_row_image_size: bool = False,
    positives_only: bool = False,
    min_tracklet_rows: int = 0,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 64,
    error_scale: float = 8.0,
    dynamics_score_mode: str = "learned_consistency",
    thresholds: list[float] | None = None,
    balance_by: list[str] | None = None,
    model_type: str = "mlp",
    diffusion_steps: int = 16,
    prior_image_size: tuple[int, int] | None = None,
    prior_sigma_scale: float = 1.5,
    prior_min_sigma: float = 2.0,
    prior_split_horizon: bool = False,
    prior_merge_mode: str = "max",
) -> ActionPolicyEvalResult:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    action_samples = out_root / "action_chunk_samples.jsonl"
    policy_weights = out_root / "action_chunk_policy.pt"
    sample_scores = out_root / "action_chunk_sample_scores.jsonl"
    tracklet_scores = out_root / "tracklet_dynamics_scores.jsonl"
    attached_tracklets = out_root / "tracklets_with_action_dynamics.jsonl"

    dataset_result = export_action_chunk_dataset_from_tracklets(
        tracklet_jsonl,
        action_samples,
        past_len=past_len,
        future_len=future_len,
        image_size=image_size,
        normalize_by_row_image_size=normalize_by_row_image_size,
        positives_only=positives_only,
        min_tracklet_rows=min_tracklet_rows,
    )
    train_action_chunk_policy(
        action_samples,
        policy_weights,
        epochs=epochs,
        lr=lr,
        hidden=hidden,
        batch_size=batch_size,
        balance_by=balance_by,
        model_type=model_type,
        diffusion_steps=diffusion_steps,
    )
    sample_eval = evaluate_action_chunk_policy(action_samples, policy_weights, sample_scores)
    tracklet_eval = score_tracklets_with_action_policy(
        tracklet_jsonl,
        policy_weights,
        tracklet_scores,
        past_len=past_len,
        future_len=future_len,
        image_size=image_size,
        normalize_by_row_image_size=normalize_by_row_image_size,
        error_scale=error_scale,
        dynamics_score_mode=dynamics_score_mode,
        min_tracklet_rows=min_tracklet_rows,
    )
    attach_eval = attach_action_dynamics_scores_to_tracklets(tracklet_jsonl, tracklet_scores, attached_tracklets)
    threshold_eval = evaluate_action_dynamics_thresholds(tracklet_scores, out_root, thresholds=thresholds)
    prior_eval = None
    frame_prior_eval = None
    frame_prior_attach_eval = None
    final_tracklets = attached_tracklets
    if prior_image_size is not None:
        prior_eval = export_action_prior_heatmaps_from_sample_scores(
            sample_scores,
            out_root / "action_priors",
            image_size=prior_image_size,
            sigma_scale=prior_sigma_scale,
            min_sigma=prior_min_sigma,
            split_horizon=prior_split_horizon,
        )
        if prior_split_horizon:
            frame_prior_eval = build_frame_prior_index_from_heatmaps(
                prior_eval.jsonl_path,
                out_root / "frame_priors",
                merge_mode=prior_merge_mode,
            )
            frame_prior_attached_tracklets = out_root / "action_dynamics_frame_prior_tracklets.jsonl"
            frame_prior_attach_eval = attach_frame_priors_to_tracklets(
                attached_tracklets,
                frame_prior_eval.jsonl_path,
                frame_prior_attached_tracklets,
            )
            final_tracklets = frame_prior_attached_tracklets

    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "out_dir": str(out_root),
        "past_len": past_len,
        "future_len": future_len,
        "image_size": list(image_size) if image_size is not None else None,
        "normalize_by_row_image_size": normalize_by_row_image_size,
        "positives_only": positives_only,
        "min_tracklet_rows": min_tracklet_rows,
        "epochs": epochs,
        "lr": lr,
        "hidden": hidden,
        "batch_size": batch_size,
        "error_scale": error_scale,
        "dynamics_score_mode": dynamics_score_mode,
        "thresholds": thresholds,
        "balance_by": balance_by,
        "model_type": model_type,
        "diffusion_steps": diffusion_steps,
        "prior_image_size": list(prior_image_size) if prior_image_size is not None else None,
        "prior_sigma_scale": prior_sigma_scale,
        "prior_min_sigma": prior_min_sigma,
        "prior_split_horizon": prior_split_horizon,
        "prior_merge_mode": prior_merge_mode,
        "action_samples": str(action_samples),
        "policy_weights": str(policy_weights),
        "sample_scores": str(sample_scores),
        "tracklet_scores": str(tracklet_scores),
        "attached_tracklets": str(attached_tracklets),
        "final_tracklets": str(final_tracklets),
        "threshold_sweep": str(threshold_eval.out_path),
        "action_prior_manifest": str(prior_eval.jsonl_path) if prior_eval is not None else None,
        "frame_prior_index": str(frame_prior_eval.jsonl_path) if frame_prior_eval is not None else None,
        "dataset": dataset_result.summary,
        "sample_eval": sample_eval.summary,
        "tracklet_eval": tracklet_eval.summary,
        "attach": attach_eval.summary,
        "threshold_eval": threshold_eval.summary,
        "action_prior_eval": prior_eval.summary if prior_eval is not None else None,
        "frame_prior_eval": frame_prior_eval.summary if frame_prior_eval is not None else None,
        "frame_prior_attach_eval": frame_prior_attach_eval.summary if frame_prior_attach_eval is not None else None,
    }
    summary_path = out_root / "action_dynamics_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=final_tracklets, summary=summary)


def run_action_dynamics_tracklet_ablation(
    tracklet_jsonl: str | Path,
    out_dir: str | Path,
    model_types: list[str] | None = None,
    past_len: int = 8,
    future_len: int = 8,
    image_size: tuple[int, int] | None = None,
    normalize_by_row_image_size: bool = False,
    positives_only: bool = False,
    min_tracklet_rows: int = 0,
    epochs: int = 50,
    lr: float = 1e-3,
    hidden: int = 128,
    batch_size: int = 64,
    error_scale: float = 8.0,
    dynamics_score_mode: str = "learned_consistency",
    thresholds: list[float] | None = None,
    balance_by: list[str] | None = None,
    diffusion_steps: int = 16,
    prior_image_size: tuple[int, int] | None = None,
    prior_sigma_scale: float = 1.5,
    prior_min_sigma: float = 2.0,
    prior_split_horizon: bool = False,
    prior_merge_mode: str = "max",
) -> ActionPolicyEvalResult:
    models = model_types or ["mlp", "residual_mlp", "diffusion"]
    if not models:
        raise ValueError("model_types must contain at least one backend")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    model_summaries: dict[str, Any] = {}
    for model_type in models:
        safe_name = model_type.replace("/", "_")
        model_dir = out_root / safe_name
        result = run_action_dynamics_tracklet_pipeline(
            tracklet_jsonl,
            model_dir,
            past_len=past_len,
            future_len=future_len,
            image_size=image_size,
            normalize_by_row_image_size=normalize_by_row_image_size,
            positives_only=positives_only,
            min_tracklet_rows=min_tracklet_rows,
            epochs=epochs,
            lr=lr,
            hidden=hidden,
            batch_size=batch_size,
            error_scale=error_scale,
            dynamics_score_mode=dynamics_score_mode,
            thresholds=thresholds,
            balance_by=balance_by,
            model_type=model_type,
            diffusion_steps=diffusion_steps,
            prior_image_size=prior_image_size,
            prior_sigma_scale=prior_sigma_scale,
            prior_min_sigma=prior_min_sigma,
            prior_split_horizon=prior_split_horizon,
            prior_merge_mode=prior_merge_mode,
        )
        best = dict(result.summary["threshold_eval"]["best"])
        tracklet_eval = dict(result.summary["tracklet_eval"])
        row = {
            "model_type": model_type,
            "out_dir": str(model_dir),
            "attached_tracklets": str(result.out_path),
            "tracklet_scores": str(result.summary["tracklet_scores"]),
            "threshold_sweep": str(result.summary["threshold_sweep"]),
            "action_prior_manifest": str(result.summary.get("action_prior_manifest") or ""),
            "frame_prior_index": str(result.summary.get("frame_prior_index") or ""),
            "scored_tracklets": int(tracklet_eval.get("scored_tracklets", 0)),
            "action_windows": int(tracklet_eval.get("action_windows", 0)),
            "mean_learned_center_error": float(tracklet_eval.get("mean_learned_center_error", 0.0)),
            "mean_constant_velocity_center_error": float(tracklet_eval.get("mean_constant_velocity_center_error", 0.0)),
            "mean_error_improvement_vs_cv": float(tracklet_eval.get("mean_error_improvement_vs_cv", 0.0)),
            "best_threshold": float(best.get("threshold", 0.0)),
            "best_precision": float(best.get("precision", 0.0)),
            "best_recall": float(best.get("recall", 0.0)),
            "best_f1": float(best.get("f1", 0.0)),
            "best_accuracy": float(best.get("accuracy", 0.0)),
            "best_tp": int(best.get("tp", 0)),
            "best_fp": int(best.get("fp", 0)),
            "best_fn": int(best.get("fn", 0)),
            "best_tn": int(best.get("tn", 0)),
            "diffusion_steps": diffusion_steps if model_type == "diffusion" else 0,
        }
        rows.append(row)
        model_summaries[model_type] = result.summary

    best_row = max(rows, key=lambda row: (row["best_f1"], row["best_recall"], row["best_precision"]))
    csv_path = out_root / "action_dynamics_tracklet_ablation.csv"
    json_path = out_root / "action_dynamics_tracklet_ablation_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "scored_tracklets",
                "action_windows",
                "mean_learned_center_error",
                "mean_constant_velocity_center_error",
                "mean_error_improvement_vs_cv",
                "best_threshold",
                "best_precision",
                "best_recall",
                "best_f1",
                "best_accuracy",
                "best_tp",
                "best_fp",
                "best_fn",
                "best_tn",
                "diffusion_steps",
                "out_dir",
                "attached_tracklets",
                "tracklet_scores",
                "threshold_sweep",
                "action_prior_manifest",
                "frame_prior_index",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "out_dir": str(out_root),
        "csv": str(csv_path),
        "json": str(json_path),
        "model_types": models,
        "past_len": past_len,
        "future_len": future_len,
        "image_size": list(image_size) if image_size is not None else None,
        "positives_only": positives_only,
        "min_tracklet_rows": min_tracklet_rows,
        "epochs": epochs,
        "lr": lr,
        "hidden": hidden,
        "batch_size": batch_size,
        "error_scale": error_scale,
        "thresholds": thresholds,
        "balance_by": balance_by,
        "diffusion_steps": diffusion_steps,
        "prior_image_size": list(prior_image_size) if prior_image_size is not None else None,
        "prior_sigma_scale": prior_sigma_scale,
        "prior_min_sigma": prior_min_sigma,
        "prior_split_horizon": prior_split_horizon,
        "prior_merge_mode": prior_merge_mode,
        "best": best_row,
        "rows": rows,
        "models": model_summaries,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=csv_path, summary=summary)


def _route_b_table_row(summary_path: Path, dataset_name: str, summary: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if "action_frame_prior_fusion" in str(summary.get("csv", "")) or summary_path.name.startswith("action_frame_prior_fusion"):
        best = dict(row or summary.get("best") or {})
        raw = dict(summary.get("raw") or {})
        return {
            "dataset": dataset_name,
            "summary_json": str(summary_path),
            "tracklet_jsonl": "",
            "run_type": "action_prior_fusion_run_sweep" if "run_roots" in summary else "action_prior_fusion_sweep",
            "model_type": "action_prior",
            "past_len": 0,
            "future_len": 0,
            "scored_tracklets": 0,
            "action_windows": 0,
            "mean_learned_center_error": 0.0,
            "mean_constant_velocity_center_error": 0.0,
            "mean_error_improvement_vs_cv": 0.0,
            "best_threshold": float(best.get("promote_threshold") or 0.0),
            "best_precision": float(best.get("precision", 0.0)),
            "best_recall": float(best.get("recall", 0.0)),
            "best_f1": float(best.get("f1", 0.0)),
            "best_accuracy": 0.0,
            "best_tp": int(best.get("tp", 0)),
            "best_fp": int(best.get("fp", 0)),
            "best_fn": int(best.get("fn", 0)),
            "best_tn": 0,
            "raw_precision": float(raw.get("precision", 0.0)),
            "raw_recall": float(raw.get("recall", 0.0)),
            "raw_f1": float(raw.get("f1", 0.0)),
            "delta_f1": float(best.get("f1", 0.0)) - float(raw.get("f1", 0.0)),
            "prior_weight": float(best.get("prior_weight", 0.0)),
            "min_prior_score": float(best.get("min_prior_score", 0.0)),
            "promote_threshold": "" if best.get("promote_threshold") is None else float(best.get("promote_threshold", 0.0)),
            "fused_rows": int(best.get("fused_rows", 0)),
            "promoted_rows": int(best.get("promoted_rows", 0)),
            "out_dir": str(summary.get("out_dir", "")),
            "tracklet_scores": "",
            "threshold_sweep": str(summary.get("csv", "")),
        }
    return {
        "dataset": dataset_name,
        "summary_json": str(summary_path),
        "tracklet_jsonl": str(summary.get("tracklet_jsonl", "")),
        "run_type": "tracklet_ablation" if "rows" in summary else "pipeline",
        "model_type": str(row.get("model_type", summary.get("model_type", ""))),
        "past_len": int(summary.get("past_len", 0)),
        "future_len": int(summary.get("future_len", 0)),
        "scored_tracklets": int(row.get("scored_tracklets", summary.get("tracklet_eval", {}).get("scored_tracklets", 0))),
        "action_windows": int(row.get("action_windows", summary.get("tracklet_eval", {}).get("action_windows", 0))),
        "mean_learned_center_error": float(
            row.get("mean_learned_center_error", summary.get("tracklet_eval", {}).get("mean_learned_center_error", 0.0))
        ),
        "mean_constant_velocity_center_error": float(
            row.get(
                "mean_constant_velocity_center_error",
                summary.get("tracklet_eval", {}).get("mean_constant_velocity_center_error", 0.0),
            )
        ),
        "mean_error_improvement_vs_cv": float(
            row.get("mean_error_improvement_vs_cv", summary.get("tracklet_eval", {}).get("mean_error_improvement_vs_cv", 0.0))
        ),
        "best_threshold": float(row.get("best_threshold", summary.get("threshold_eval", {}).get("best", {}).get("threshold", 0.0))),
        "best_precision": float(row.get("best_precision", summary.get("threshold_eval", {}).get("best", {}).get("precision", 0.0))),
        "best_recall": float(row.get("best_recall", summary.get("threshold_eval", {}).get("best", {}).get("recall", 0.0))),
        "best_f1": float(row.get("best_f1", summary.get("threshold_eval", {}).get("best", {}).get("f1", 0.0))),
        "best_accuracy": float(row.get("best_accuracy", summary.get("threshold_eval", {}).get("best", {}).get("accuracy", 0.0))),
        "best_tp": int(row.get("best_tp", summary.get("threshold_eval", {}).get("best", {}).get("tp", 0))),
        "best_fp": int(row.get("best_fp", summary.get("threshold_eval", {}).get("best", {}).get("fp", 0))),
        "best_fn": int(row.get("best_fn", summary.get("threshold_eval", {}).get("best", {}).get("fn", 0))),
        "best_tn": int(row.get("best_tn", summary.get("threshold_eval", {}).get("best", {}).get("tn", 0))),
        "out_dir": str(row.get("out_dir", summary.get("out_dir", ""))),
        "tracklet_scores": str(row.get("tracklet_scores", summary.get("tracklet_scores", ""))),
        "threshold_sweep": str(row.get("threshold_sweep", summary.get("threshold_sweep", ""))),
    }


def collect_route_b_result_summaries(
    summaries: list[str | Path],
    out_dir: str | Path,
    dataset_names: list[str] | None = None,
) -> ActionPolicyEvalResult:
    if not summaries:
        raise ValueError("summaries must contain at least one Route B summary JSON")
    if dataset_names is not None and len(dataset_names) != len(summaries):
        raise ValueError("dataset_names must have the same length as summaries")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, summary_file in enumerate(summaries):
        summary_path = Path(summary_file)
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        dataset_name = dataset_names[index] if dataset_names is not None else summary_path.parent.name
        if "best" in summary and ("action_frame_prior_fusion" in str(summary.get("csv", "")) or summary_path.name.startswith("action_frame_prior_fusion")):
            rows.append(_route_b_table_row(summary_path, dataset_name, summary, dict(summary.get("best") or {})))
        elif "rows" in summary:
            for row in summary["rows"]:
                rows.append(_route_b_table_row(summary_path, dataset_name, summary, dict(row)))
        else:
            rows.append(_route_b_table_row(summary_path, dataset_name, summary, {}))

    if not rows:
        raise ValueError("No Route B rows could be extracted from summaries")
    best = max(rows, key=lambda row: (row["best_f1"], row["best_recall"], row["best_precision"]))
    csv_path = out_root / "route_b_results_table.csv"
    json_path = out_root / "route_b_results_summary.json"
    fieldnames = [
        "dataset",
        "run_type",
        "model_type",
        "past_len",
        "future_len",
        "scored_tracklets",
        "action_windows",
        "mean_learned_center_error",
        "mean_constant_velocity_center_error",
        "mean_error_improvement_vs_cv",
        "best_threshold",
        "best_precision",
        "best_recall",
        "best_f1",
        "best_accuracy",
        "best_tp",
        "best_fp",
        "best_fn",
        "best_tn",
        "summary_json",
        "tracklet_jsonl",
        "out_dir",
        "tracklet_scores",
        "threshold_sweep",
        "raw_precision",
        "raw_recall",
        "raw_f1",
        "delta_f1",
        "prior_weight",
        "min_prior_score",
        "promote_threshold",
        "fused_rows",
        "promoted_rows",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "summaries": [str(path) for path in summaries],
        "out_dir": str(out_root),
        "csv": str(csv_path),
        "json": str(json_path),
        "num_rows": len(rows),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "model_types": sorted({str(row["model_type"]) for row in rows}),
        "best": best,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=csv_path, summary=summary)


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _float_field(row: dict[str, Any], field: str, default: float = 0.0) -> float:
    value = row.get(field, default)
    if value in (None, ""):
        return default
    return float(value)


def compare_route_b_results_to_baselines(
    route_b_csv: str | Path,
    baseline_csv: str | Path,
    out_dir: str | Path,
    metric: str = "best_f1",
    higher_is_better: bool = True,
) -> ActionPolicyEvalResult:
    route_rows = _read_csv_rows(route_b_csv)
    baseline_rows = _read_csv_rows(baseline_csv)
    if not route_rows:
        raise ValueError("route_b_csv is empty")
    if not baseline_rows:
        raise ValueError("baseline_csv is empty")

    combined_rows: list[dict[str, Any]] = []
    for row in baseline_rows:
        dataset = str(row.get("dataset", ""))
        method = str(row.get("method") or row.get("model_type") or row.get("name") or "baseline")
        combined_rows.append(
            {
                "dataset": dataset,
                "source": "baseline",
                "method": method,
                "model_type": str(row.get("model_type", "")),
                "metric": metric,
                "metric_value": _float_field(row, metric),
                "best_precision": _float_field(row, "best_precision", _float_field(row, "precision")),
                "best_recall": _float_field(row, "best_recall", _float_field(row, "recall")),
                "best_f1": _float_field(row, "best_f1", _float_field(row, "f1")),
                "summary_json": str(row.get("summary_json", "")),
            }
        )
    for row in route_rows:
        dataset = str(row.get("dataset", ""))
        model_type = str(row.get("model_type", "route_b"))
        combined_rows.append(
            {
                "dataset": dataset,
                "source": "route_b",
                "method": f"route_b:{model_type}",
                "model_type": model_type,
                "metric": metric,
                "metric_value": _float_field(row, metric),
                "best_precision": _float_field(row, "best_precision"),
                "best_recall": _float_field(row, "best_recall"),
                "best_f1": _float_field(row, "best_f1"),
                "summary_json": str(row.get("summary_json", "")),
            }
        )

    reverse = bool(higher_is_better)
    datasets = sorted({str(row["dataset"]) for row in combined_rows})
    comparison_rows = []
    ranking_rows = []
    for dataset in datasets:
        dataset_rows = [row for row in combined_rows if row["dataset"] == dataset]
        ranked = sorted(dataset_rows, key=lambda row: float(row["metric_value"]), reverse=reverse)
        for rank, row in enumerate(ranked, start=1):
            ranking_rows.append({**row, "rank": rank})
        baseline_candidates = [row for row in dataset_rows if row["source"] == "baseline"]
        route_candidates = [row for row in dataset_rows if row["source"] == "route_b"]
        if not baseline_candidates or not route_candidates:
            continue
        best_baseline = sorted(baseline_candidates, key=lambda row: float(row["metric_value"]), reverse=reverse)[0]
        best_route_b = sorted(route_candidates, key=lambda row: float(row["metric_value"]), reverse=reverse)[0]
        delta = float(best_route_b["metric_value"]) - float(best_baseline["metric_value"])
        if not higher_is_better:
            delta = -delta
        comparison_rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "best_baseline_method": best_baseline["method"],
                "best_baseline_value": float(best_baseline["metric_value"]),
                "best_route_b_method": best_route_b["method"],
                "best_route_b_value": float(best_route_b["metric_value"]),
                "delta_route_b_minus_baseline": delta,
                "route_b_beats_baseline": delta > 0,
            }
        )

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    comparison_csv = out_root / "route_b_baseline_comparison.csv"
    ranking_csv = out_root / "route_b_baseline_ranking.csv"
    json_path = out_root / "route_b_baseline_comparison_summary.json"
    with comparison_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "metric",
                "higher_is_better",
                "best_baseline_method",
                "best_baseline_value",
                "best_route_b_method",
                "best_route_b_value",
                "delta_route_b_minus_baseline",
                "route_b_beats_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)
    with ranking_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "rank",
                "source",
                "method",
                "model_type",
                "metric",
                "metric_value",
                "best_precision",
                "best_recall",
                "best_f1",
                "summary_json",
            ],
        )
        writer.writeheader()
        writer.writerows(ranking_rows)

    summary = {
        "route_b_csv": str(route_b_csv),
        "baseline_csv": str(baseline_csv),
        "out_dir": str(out_root),
        "comparison_csv": str(comparison_csv),
        "ranking_csv": str(ranking_csv),
        "json": str(json_path),
        "metric": metric,
        "higher_is_better": higher_is_better,
        "datasets": datasets,
        "num_comparisons": len(comparison_rows),
        "route_b_wins": int(sum(1 for row in comparison_rows if row["route_b_beats_baseline"])),
        "comparison_rows": comparison_rows,
        "ranking_rows": ranking_rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=comparison_csv, summary=summary)


def write_route_b_baseline_template(
    out: str | Path,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
) -> ActionPolicyEvalResult:
    dataset_names = datasets or ["nps", "aot", "transvisdrone"]
    method_names = methods or ["NPS-paper", "YOLOMG", "TransVisDrone"]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "best_precision",
        "best_recall",
        "best_f1",
        "best_accuracy",
        "paper_url",
        "source_notes",
    ]
    rows = []
    for dataset in dataset_names:
        for method in method_names:
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "best_precision": "",
                    "best_recall": "",
                    "best_f1": "",
                    "best_accuracy": "",
                    "paper_url": "",
                    "source_notes": "Fill with paper/table/split details before compare-route-b-baselines",
                }
            )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "csv": str(out_path),
        "datasets": dataset_names,
        "methods": method_names,
        "rows": len(rows),
        "required_for_compare": ["dataset", "method", "best_f1"],
        "optional_metrics": ["best_precision", "best_recall", "best_accuracy"],
    }
    return ActionPolicyEvalResult(out_path=out_path, summary=summary)


def _f1_from_pr(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def write_route_b_official_baseline_seed(
    out: str | Path,
    include_placeholders: bool = True,
) -> ActionPolicyEvalResult:
    """Write a provenance-heavy baseline seed CSV for official Route B reports.

    Empty metric cells are intentional placeholders. Run validate-route-b-baselines
    with --allow-empty-metric while drafting, then fill the target metric before
    using the CSV for a strict comparison.
    """
    source_backed_rows = [
        {
            "dataset": "nps",
            "method": "TransVisDrone",
            "protocol": "NPS val, repo protocol, best weights",
            "best_precision": 0.901,
            "best_recall": 0.881,
            "best_f1": _f1_from_pr(0.901, 0.881),
            "best_accuracy": "",
            "nps_map50": 0.948,
            "nps_map5095": 0.464,
            "aot_hfar": "",
            "aot_fppi": "",
            "aot_edr300_detection": "",
            "aot_edr300_tracking": "",
            "source_path": "doc/progress_report_for_professor_en.md:103",
            "source_notes": "TVD NPS val row: P=0.901, R=0.881, mAP@0.5=0.948; best_f1 computed from P/R for Route B tooling only.",
            "needs_fill": "no",
        },
        {
            "dataset": "aot",
            "method": "TransVisDrone",
            "protocol": "AOT fulltest, conf=0.2, official airborne metrics",
            "best_precision": "",
            "best_recall": "",
            "best_f1": "",
            "best_accuracy": "",
            "nps_map50": "",
            "nps_map5095": "",
            "aot_hfar": 89.476744,
            "aot_fppi": 0.262318,
            "aot_edr300_detection": 0.925714,
            "aot_edr300_tracking": 0.925714,
            "source_path": "doc/progress_report_for_professor_en.md:117",
            "source_notes": "TVD AOT fulltest row: HFAR=89.476744, FPPI=0.262318, EDR@300 detection/tracking=0.925714.",
            "needs_fill": "metric-dependent",
        },
        {
            "dataset": "aot",
            "method": "Winner-v022",
            "protocol": "AOT fulltest, official airborne metrics",
            "best_precision": "",
            "best_recall": "",
            "best_f1": "",
            "best_accuracy": "",
            "nps_map50": "",
            "nps_map5095": "",
            "aot_hfar": 0.523,
            "aot_fppi": 0.0000146,
            "aot_edr300_detection": 0.989,
            "aot_edr300_tracking": 0.989,
            "source_path": "doc/progress_report_for_professor_en.md:128",
            "source_notes": "AOT challenge winner is included as a strong low-FP reference; exact fulltest summary path is cited in the progress report.",
            "needs_fill": "metric-dependent",
        },
    ]
    placeholder_rows = [
        {
            "dataset": "nps",
            "method": "NPS-paper",
            "protocol": "Li-TETC original NPS protocol",
            "best_precision": "",
            "best_recall": "",
            "best_f1": "",
            "best_accuracy": "",
            "nps_map50": "",
            "nps_map5095": "",
            "aot_hfar": "",
            "aot_fppi": "",
            "aot_edr300_detection": "",
            "aot_edr300_tracking": "",
            "source_path": "Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking/pt_pipeline/runs/eval_v1.json",
            "source_notes": "Fill after confirming the same NPS annotation protocol and target metric.",
            "needs_fill": "yes",
        },
        {
            "dataset": "nps",
            "method": "YOLOMG",
            "protocol": "Route B comparison target",
            "best_precision": "",
            "best_recall": "",
            "best_f1": "",
            "best_accuracy": "",
            "nps_map50": "",
            "nps_map5095": "",
            "aot_hfar": "",
            "aot_fppi": "",
            "aot_edr300_detection": "",
            "aot_edr300_tracking": "",
            "source_path": "",
            "source_notes": "Fill from the matched YOLOMG run on the same split used by Route B.",
            "needs_fill": "yes",
        },
        {
            "dataset": "aot",
            "method": "YOLOMG",
            "protocol": "Route B comparison target",
            "best_precision": "",
            "best_recall": "",
            "best_f1": "",
            "best_accuracy": "",
            "nps_map50": "",
            "nps_map5095": "",
            "aot_hfar": "",
            "aot_fppi": "",
            "aot_edr300_detection": "",
            "aot_edr300_tracking": "",
            "source_path": "",
            "source_notes": "Fill from the matched YOLOMG run on the same split used by Route B.",
            "needs_fill": "yes",
        },
    ]
    rows = source_backed_rows + (placeholder_rows if include_placeholders else [])
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "method",
        "protocol",
        "best_precision",
        "best_recall",
        "best_f1",
        "best_accuracy",
        "nps_map50",
        "nps_map5095",
        "aot_hfar",
        "aot_fppi",
        "aot_edr300_detection",
        "aot_edr300_tracking",
        "source_path",
        "source_notes",
        "needs_fill",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "csv": str(out_path),
        "rows": len(rows),
        "source_backed_rows": len(source_backed_rows),
        "placeholder_rows": len(placeholder_rows) if include_placeholders else 0,
        "include_placeholders": include_placeholders,
        "strict_compare_ready": not include_placeholders,
        "metrics": ["best_f1", "nps_map50", "aot_hfar", "aot_fppi", "aot_edr300_detection", "aot_edr300_tracking"],
        "notes": [
            "Use one metric at a time when calling compare/build-route-b-report.",
            "For AOT false-alarm metrics such as aot_hfar or aot_fppi, pass --lower-is-better.",
            "Rows marked needs_fill=yes must be completed before strict official reporting.",
        ],
    }
    return ActionPolicyEvalResult(out_path=out_path, summary=summary)


def validate_route_b_baseline_csv(
    baseline_csv: str | Path,
    out: str | Path,
    metric: str = "best_f1",
    require_metric_values: bool = True,
) -> ActionPolicyEvalResult:
    rows = _read_csv_rows(baseline_csv)
    required_columns = ["dataset", "method", metric]
    present_columns = set(rows[0].keys()) if rows else set()
    issues = []
    warnings = []
    for column in required_columns:
        if column not in present_columns:
            issues.append({"row": 0, "field": column, "severity": "error", "message": f"Missing required column: {column}"})
    seen: set[tuple[str, str]] = set()
    if rows and not issues:
        for index, row in enumerate(rows, start=2):
            dataset = str(row.get("dataset", "")).strip()
            method = str(row.get("method", "")).strip()
            key = (dataset, method)
            if not dataset:
                issues.append({"row": index, "field": "dataset", "severity": "error", "message": "dataset is empty"})
            if not method:
                issues.append({"row": index, "field": "method", "severity": "error", "message": "method is empty"})
            if dataset and method:
                if key in seen:
                    issues.append({"row": index, "field": "method", "severity": "error", "message": f"duplicate dataset-method: {dataset}/{method}"})
                seen.add(key)
            metric_value = str(row.get(metric, "")).strip()
            if not metric_value:
                target = issues if require_metric_values else warnings
                target.append({"row": index, "field": metric, "severity": "error" if require_metric_values else "warning", "message": f"{metric} is empty"})
            else:
                try:
                    value = float(metric_value)
                    if metric in {"best_precision", "best_recall", "best_f1", "best_accuracy", "precision", "recall", "f1", "accuracy"}:
                        if value < 0.0 or value > 1.0:
                            issues.append({"row": index, "field": metric, "severity": "error", "message": f"{metric} must be in [0, 1]"})
                except ValueError:
                    issues.append({"row": index, "field": metric, "severity": "error", "message": f"{metric} is not numeric"})
            for optional_metric in ["best_precision", "best_recall", "best_accuracy"]:
                value_text = str(row.get(optional_metric, "")).strip()
                if not value_text:
                    continue
                try:
                    value = float(value_text)
                    if value < 0.0 or value > 1.0:
                        issues.append({"row": index, "field": optional_metric, "severity": "error", "message": f"{optional_metric} must be in [0, 1]"})
                except ValueError:
                    issues.append({"row": index, "field": optional_metric, "severity": "error", "message": f"{optional_metric} is not numeric"})
    if not rows:
        issues.append({"row": 0, "field": "", "severity": "error", "message": "baseline CSV has no data rows"})

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline_csv": str(baseline_csv),
        "out": str(out_path),
        "metric": metric,
        "require_metric_values": require_metric_values,
        "rows": len(rows),
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=out_path, summary=summary)


def _format_metric(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def export_route_b_baseline_markdown_table(
    comparison_summary_json: str | Path,
    out: str | Path,
    digits: int = 3,
) -> ActionPolicyEvalResult:
    summary_in = json.loads(Path(comparison_summary_json).read_text(encoding="utf-8"))
    rows = list(summary_in.get("comparison_rows") or [])
    if not rows:
        raise ValueError("comparison summary has no comparison_rows")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Dataset | Best baseline | Baseline | Best Route B | Route B | Delta | Beats baseline |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        marker = "yes" if bool(row.get("route_b_beats_baseline", False)) else "no"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("dataset", "")),
                    str(row.get("best_baseline_method", "")),
                    _format_metric(row.get("best_baseline_value", 0.0), digits),
                    str(row.get("best_route_b_method", "")),
                    _format_metric(row.get("best_route_b_value", 0.0), digits),
                    _format_metric(row.get("delta_route_b_minus_baseline", 0.0), digits),
                    marker,
                ]
            )
            + " |"
        )
    metric = str(summary_in.get("metric", "best_f1"))
    wins = int(summary_in.get("route_b_wins", 0))
    total = int(summary_in.get("num_comparisons", len(rows)))
    lines.extend(
        [
            "",
            f"Caption: Route B action-dynamics results compared with published baselines using `{metric}`.",
            f"Route B beats the best listed baseline on {wins}/{total} dataset comparisons.",
            "Note: verify that all baseline entries use the same split, threshold protocol, and metric definition before reporting.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result_summary = {
        "comparison_summary_json": str(comparison_summary_json),
        "out": str(out_path),
        "metric": metric,
        "rows": len(rows),
        "route_b_wins": wins,
        "num_comparisons": total,
    }
    return ActionPolicyEvalResult(out_path=out_path, summary=result_summary)


def build_route_b_baseline_report(
    summaries: list[str | Path],
    baseline_csv: str | Path,
    out_dir: str | Path,
    dataset_names: list[str] | None = None,
    metric: str = "best_f1",
    higher_is_better: bool = True,
    digits: int = 3,
    strict_baselines: bool = True,
) -> ActionPolicyEvalResult:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    collect = collect_route_b_result_summaries(summaries, out_root / "collected", dataset_names=dataset_names)
    validation = validate_route_b_baseline_csv(
        baseline_csv,
        out_root / "baseline_validation.json",
        metric=metric,
        require_metric_values=True,
    )
    if strict_baselines and not validation.summary["valid"]:
        summary = {
            "summaries": [str(path) for path in summaries],
            "baseline_csv": str(baseline_csv),
            "out_dir": str(out_root),
            "metric": metric,
            "strict_baselines": strict_baselines,
            "valid": False,
            "route_b_results_csv": str(collect.out_path),
            "baseline_validation_json": str(validation.out_path),
            "baseline_validation": validation.summary,
        }
        report_json = out_root / "route_b_report_summary.json"
        report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return ActionPolicyEvalResult(out_path=report_json, summary=summary)

    comparison = compare_route_b_results_to_baselines(
        collect.out_path,
        baseline_csv,
        out_root / "comparison",
        metric=metric,
        higher_is_better=higher_is_better,
    )
    markdown = export_route_b_baseline_markdown_table(
        comparison.summary["json"],
        out_root / "route_b_baseline_report.md",
        digits=digits,
    )
    summary = {
        "summaries": [str(path) for path in summaries],
        "baseline_csv": str(baseline_csv),
        "out_dir": str(out_root),
        "metric": metric,
        "higher_is_better": higher_is_better,
        "digits": digits,
        "strict_baselines": strict_baselines,
        "valid": bool(validation.summary["valid"]),
        "route_b_results_csv": str(collect.out_path),
        "route_b_results_json": str(collect.summary["json"]),
        "baseline_validation_json": str(validation.out_path),
        "comparison_csv": str(comparison.out_path),
        "comparison_json": str(comparison.summary["json"]),
        "ranking_csv": str(comparison.summary["ranking_csv"]),
        "markdown": str(markdown.out_path),
        "route_b_wins": int(comparison.summary["route_b_wins"]),
        "num_comparisons": int(comparison.summary["num_comparisons"]),
        "best_route_b": collect.summary["best"],
        "comparison_rows": comparison.summary["comparison_rows"],
    }
    report_json = out_root / "route_b_report_summary.json"
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=markdown.out_path, summary=summary)


def evaluate_action_dynamics_thresholds(
    dynamics_scores_jsonl: str | Path,
    out_dir: str | Path,
    thresholds: list[float] | None = None,
) -> ActionPolicyEvalResult:
    rows = []
    with Path(dynamics_scores_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("Dynamics score JSONL is empty")
    labels = np.asarray([int(float(row.get("label", 0))) > 0 for row in rows], dtype=bool)
    scores = np.asarray([float(row.get("dynamics_score", 0.0)) for row in rows], dtype=np.float32)
    if thresholds is None or not thresholds:
        thresholds = sorted({0.0, 0.25, 0.5, 0.75, 1.0, *[float(v) for v in scores.tolist()]})
    sweep_rows = []
    for threshold in thresholds:
        pred = scores >= float(threshold)
        tp = int(np.sum(pred & labels))
        fp = int(np.sum(pred & ~labels))
        fn = int(np.sum(~pred & labels))
        tn = int(np.sum(~pred & ~labels))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        accuracy = (tp + tn) / max(1, len(labels))
        sweep_rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(accuracy),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    best = max(sweep_rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "action_dynamics_threshold_sweep.csv"
    json_path = out_path / "action_dynamics_threshold_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"])
        writer.writeheader()
        writer.writerows(sweep_rows)
    summary = {
        "dynamics_scores_jsonl": str(dynamics_scores_jsonl),
        "out_dir": str(out_path),
        "csv": str(csv_path),
        "json": str(json_path),
        "num_tracklets": int(len(rows)),
        "positives": int(np.sum(labels)),
        "negatives": int(np.sum(~labels)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "best": best,
        "thresholds": sweep_rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ActionPolicyEvalResult(out_path=csv_path, summary=summary)
