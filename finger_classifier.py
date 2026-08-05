"""Small RandomForest model for keydown-window finger classification."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from calibration_collector import FEATURE_NAMES, SCHEMA_VERSION
from keyboard_layouts import normalize_key_label


DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("finger_model.pkl")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    samples: int
    classes: int
    training_accuracy: float
    model_path: Path


class FingerClassifier:
    def __init__(
        self,
        estimator: RandomForestClassifier,
        key_regions: dict[str, tuple[float, float]],
        feature_count: int,
    ) -> None:
        self.estimator = estimator
        self.key_regions = key_regions
        self.feature_count = feature_count

    def predict_one(self, features: np.ndarray) -> tuple[str, float]:
        vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.feature_count:
            raise ValueError(
                f"model expects {self.feature_count} features, got {vector.shape[1]}"
            )
        probabilities = self.estimator.predict_proba(vector)[0]
        index = int(np.argmax(probabilities))
        return str(self.estimator.classes_[index]), float(probabilities[index])

    def target_for_key(self, key_label: str) -> tuple[float, float] | None:
        return self.key_regions.get(normalize_key_label(key_label))

    def save(self, path: Path = DEFAULT_MODEL_PATH) -> None:
        path = Path(path)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "sklearn_version": sklearn.__version__,
            "created_unix_ns": time.time_ns(),
            "feature_count": self.feature_count,
            "feature_names": FEATURE_NAMES,
            "key_regions": self.key_regions,
            "estimator": self.estimator,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "FingerClassifier":
        # Pickle is intentionally local-only. Never load an artifact from an untrusted source.
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("model feature schema is incompatible")
        if payload.get("feature_names") != FEATURE_NAMES:
            raise ValueError("model feature order is incompatible")
        trained_version = str(payload.get("sklearn_version", ""))
        if trained_version != sklearn.__version__:
            raise ValueError(
                f"model uses scikit-learn {trained_version}; runtime has {sklearn.__version__}"
            )
        return cls(
            payload["estimator"],
            {key: tuple(value) for key, value in payload["key_regions"].items()},
            int(payload["feature_count"]),
        )


def train_model(
    data_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> TrainingResult:
    payload = json.loads(Path(data_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("calibration dataset schema is incompatible")
    if payload.get("feature_names") != FEATURE_NAMES:
        raise ValueError("calibration feature order is incompatible")
    samples = payload.get("samples", [])
    if len(samples) < 10:
        raise ValueError("at least 10 accepted calibration samples are required")

    x = np.asarray([sample["features"] for sample in samples], dtype=np.float32)
    y = np.asarray([sample["label"] for sample in samples])
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("calibration must contain at least two finger classes")
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("calibration feature matrix has the wrong shape")

    estimator = RandomForestClassifier(
        n_estimators=180,
        max_depth=14,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    estimator.fit(x, y)

    region_points: dict[str, list[tuple[float, float]]] = {}
    for sample in samples:
        point = sample.get("target_fingertip_xy")
        if point is None or len(point) != 2:
            continue
        region_points.setdefault(normalize_key_label(sample["key"]), []).append(
            (float(point[0]), float(point[1]))
        )
    key_regions = {
        key: tuple(np.median(np.asarray(points), axis=0).tolist())
        for key, points in region_points.items()
    }

    classifier = FingerClassifier(estimator, key_regions, x.shape[1])
    classifier.save(model_path)
    accuracy = float(estimator.score(x, y))
    return TrainingResult(len(samples), int(classes.size), accuracy, Path(model_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    result = train_model(args.data, args.output)
    print(
        f"Saved {result.model_path} from {result.samples} samples across "
        f"{result.classes} classes (training accuracy {result.training_accuracy:.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
