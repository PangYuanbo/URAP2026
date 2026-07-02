from __future__ import annotations

import argparse
import os
import re
import runpy
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TEST_IND_RE = re.compile(r"(?m)^(\s*)test_ind = index\[10\*\(ind-1\):10\*\(ind-1\)\+1\]\s*$")


class LegacyDecisionTree:
    """Small runtime for sklearn 0.19 tree pickles used by Li-TETC."""

    def __init__(self, *args):
        self.args = args

    def __setstate__(self, state):
        self.__dict__.update(state)

    def apply(self, x):
        x = np.asarray(x)
        leaves = np.empty(x.shape[0], dtype=np.int64)
        for row_index, row in enumerate(x):
            node_index = 0
            while True:
                node = self.nodes[node_index]
                left = int(node["left_child"])
                right = int(node["right_child"])
                feature = int(node["feature"])
                if left == -1 and right == -1:
                    leaves[row_index] = node_index
                    break
                node_index = left if row[feature] <= float(node["threshold"]) else right
        return leaves

    def predict_proba(self, x):
        values = np.asarray(self.values[self.apply(x), 0, :], dtype=float)
        denom = values.sum(axis=1, keepdims=True)
        return np.divide(values, denom, out=np.zeros_like(values), where=denom > 0)


class LegacyDecisionTreeClassifier:
    def __setstate__(self, state):
        self.__dict__.update(state)

    def predict_proba(self, x):
        return self.tree_.predict_proba(x)

    def predict(self, x):
        proba = self.predict_proba(x)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)


class LegacyAdaBoostClassifier:
    def __setstate__(self, state):
        self.__dict__.update(state)

    @staticmethod
    def _samme_proba(estimator, n_classes, x):
        proba = estimator.predict_proba(x)
        eps = np.finfo(proba.dtype).eps
        proba = np.clip(proba, eps, None)
        log_proba = np.log(proba)
        return (n_classes - 1) * (log_proba - (1.0 / n_classes) * log_proba.sum(axis=1)[:, np.newaxis])

    def decision_function(self, x):
        n_classes = int(self.n_classes_)
        weights = np.asarray(self.estimator_weights_, dtype=float)
        if str(getattr(self, "algorithm", "")) == "SAMME.R":
            scores = sum(
                weight * self._samme_proba(estimator, n_classes, x)
                for estimator, weight in zip(self.estimators_, weights)
            )
        else:
            scores = sum(
                weight * np.eye(n_classes)[np.searchsorted(self.classes_, estimator.predict(x))]
                for estimator, weight in zip(self.estimators_, weights)
            )
        scores /= max(float(weights.sum()), np.finfo(float).eps)
        return scores[:, 1] - scores[:, 0] if n_classes == 2 else scores

    def predict(self, x):
        decision = self.decision_function(x)
        if int(self.n_classes_) == 2:
            return self.classes_.take(decision > 0, axis=0)
        return self.classes_.take(np.argmax(decision, axis=1), axis=0)


def install_compat_shims() -> None:
    import cv2
    import joblib
    import joblib.numpy_pickle as joblib_numpy_pickle
    import sklearn.externals

    original_joblib_load = joblib.load

    def legacy_joblib_load(filename, *args, **kwargs):
        backups = {name: sys.modules.get(name) for name in (
            "sklearn.externals.joblib.numpy_pickle",
            "sklearn.ensemble.weight_boosting",
            "sklearn.tree.tree",
            "sklearn.tree._tree",
        )}
        try:
            sys.modules["sklearn.externals.joblib.numpy_pickle"] = joblib_numpy_pickle

            ensemble_legacy = types.ModuleType("sklearn.ensemble.weight_boosting")
            ensemble_legacy.AdaBoostClassifier = LegacyAdaBoostClassifier
            sys.modules["sklearn.ensemble.weight_boosting"] = ensemble_legacy

            tree_legacy = types.ModuleType("sklearn.tree.tree")
            tree_legacy.DecisionTreeClassifier = LegacyDecisionTreeClassifier
            sys.modules["sklearn.tree.tree"] = tree_legacy

            tree_core_legacy = types.ModuleType("sklearn.tree._tree")
            tree_core_legacy.Tree = LegacyDecisionTree
            sys.modules["sklearn.tree._tree"] = tree_core_legacy

            return original_joblib_load(filename, *args, **kwargs)
        finally:
            for name, module in backups.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    sklearn.externals.joblib = joblib
    sys.modules["sklearn.externals.joblib"] = joblib
    joblib.load = legacy_joblib_load

    # The original code targets Python 3.7 / NumPy 1.16 era aliases.
    for name, value in {
        "int": int,
        "float": float,
        "bool": bool,
        "object": object,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)

    def int_point(point):
        return tuple(int(round(float(value))) for value in point)

    original_circle = cv2.circle
    original_rectangle = cv2.rectangle
    original_line = cv2.line
    original_put_text = cv2.putText

    def circle(img, center, radius, color, thickness=None, *args, **kwargs):
        return original_circle(img, int_point(center), int(round(float(radius))), color, thickness if thickness is not None else 1, *args, **kwargs)

    def rectangle(img, pt1, pt2, color, thickness=None, *args, **kwargs):
        return original_rectangle(img, int_point(pt1), int_point(pt2), color, thickness if thickness is not None else 1, *args, **kwargs)

    def line(img, pt1, pt2, color, thickness=None, *args, **kwargs):
        return original_line(img, int_point(pt1), int_point(pt2), color, thickness if thickness is not None else 1, *args, **kwargs)

    def put_text(img, text, org, *args, **kwargs):
        return original_put_text(img, text, int_point(org), *args, **kwargs)

    cv2.circle = circle
    cv2.rectangle = rectangle
    cv2.line = line
    cv2.putText = put_text


def patch_main_source(source: str, video_ids: list[int] | None = None) -> str:
    if not video_ids:
        return source
    zero_based = [str(video_id - 1) for video_id in video_ids]
    replacement = r"\1test_ind = [" + ", ".join(zero_based) + "]"
    patched, count = TEST_IND_RE.subn(replacement, source)
    if count != 1:
        raise RuntimeError("Could not patch Li-TETC main.py test_ind selection.")
    return patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Li-TETC demo with compatibility shims for newer Python packages.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=ROOT / "papers" / "Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking",
    )
    parser.add_argument(
        "--video-id",
        type=int,
        action="append",
        default=[],
        help="1-based Li-TETC video id to run, e.g. 14 or 40. Can be repeated. Default preserves the paper's original hardcoded selection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    main_py = repo / "main.py"
    if not main_py.is_file():
        raise SystemExit(f"main.py not found: {main_py}")

    install_compat_shims()
    os.chdir(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if args.video_id:
        source = patch_main_source(main_py.read_text(encoding="utf-8"), args.video_id)
        globals_dict = {"__file__": str(main_py), "__name__": "__main__", "__package__": None}
        exec(compile(source, str(main_py), "exec"), globals_dict)
    else:
        runpy.run_path(str(main_py), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
