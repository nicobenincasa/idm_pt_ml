#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models/nb4/"

MODEL_FILE = MODELS_DIR / "final_mlp_regressor.pt"
FEATURE_SCALER_FILE = MODELS_DIR / "final_feature_scaler.pkl"
TARGET_SCALER_FILE = MODELS_DIR / "final_target_scaler.pkl"
METADATA_FILE = PROJECT_ROOT / "model_metadata.json"


class PTRegressor(nn.Module):
    """MLP architecture used for the final surrogate."""

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict IDM one-step FOPT phase-transition observables."
    )

    parser.add_argument("mH", type=float)
    parser.add_argument("mA", type=float)
    parser.add_argument("mHp", type=float)
    parser.add_argument("l2", type=float)
    parser.add_argument("l345", type=float)

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print predictions and warnings as JSON.",
    )

    return parser.parse_args()


def check_required_files() -> None:
    required = [
        MODEL_FILE,
        FEATURE_SCALER_FILE,
        TARGET_SCALER_FILE,
        METADATA_FILE,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        print("Error: required file(s) are missing:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print(
            "\nRun the final-model training notebook first.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def load_surrogate():
    feature_scaler = joblib.load(FEATURE_SCALER_FILE)
    target_scaler = joblib.load(TARGET_SCALER_FILE)

    with METADATA_FILE.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    expected_inputs = ["mH", "mA", "mHp", "l2", "l345"]
    expected_targets = ["Tcrit", "Tnuc", "log_alpha", "log_beta/H"]

    if metadata.get("input_features") != expected_inputs:
        raise ValueError("Metadata input feature order does not match predict.py.")

    if metadata.get("target_columns") != expected_targets:
        raise ValueError("Metadata target order does not match predict.py.")

    model = PTRegressor(n_inputs=5, n_outputs=4)

    state_dict = torch.load(
        MODEL_FILE,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, feature_scaler, target_scaler, metadata


def check_training_domain(
    parameters: dict[str, float],
    metadata: dict,
) -> list[str]:
    """Warn when any input lies outside the min/max training range."""

    feature_ranges = metadata.get("feature_ranges")

    if feature_ranges is None:
        raise ValueError(
            "model_metadata.json does not contain 'feature_ranges'. "
            "Add the feature_ranges block to the final-model training notebook "
            "and regenerate model_metadata.json."
        )

    warnings = []

    for feature, value in parameters.items():
        if feature not in feature_ranges:
            raise ValueError(
                f"Training range for feature '{feature}' is missing from metadata."
            )

        bounds = feature_ranges[feature]
        minimum = float(bounds["min"])
        maximum = float(bounds["max"])

        if value < minimum:
            warnings.append(
                f"{feature} = {value:.8g} is below the training range "
                f"[{minimum:.8g}, {maximum:.8g}]."
            )
        elif value > maximum:
            warnings.append(
                f"{feature} = {value:.8g} is above the training range "
                f"[{minimum:.8g}, {maximum:.8g}]."
            )

    return warnings


def predict(
    model: nn.Module,
    feature_scaler,
    target_scaler,
    parameters: list[float],
) -> dict[str, float]:
    x = np.asarray(parameters, dtype=np.float32).reshape(1, -1)

    if x.shape != (1, 5):
        raise ValueError(
            "Exactly five input parameters are required: "
            "mH mA mHp l2 l345"
        )

    if not np.isfinite(x).all():
        raise ValueError("All input parameters must be finite numbers.")

    x_scaled = feature_scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_scaled = model(x_tensor).cpu().numpy()

    y_transformed = target_scaler.inverse_transform(y_scaled)[0]

    return {
        "Tcrit": float(y_transformed[0]),
        "Tnuc": float(y_transformed[1]),
        "alpha": float(10.0 ** y_transformed[2]),
        "beta/H": float(10.0 ** y_transformed[3]),
    }


def main() -> None:
    args = parse_args()
    check_required_files()

    model, feature_scaler, target_scaler, metadata = load_surrogate()

    parameters = {
        "mH": args.mH,
        "mA": args.mA,
        "mHp": args.mHp,
        "l2": args.l2,
        "l345": args.l345,
    }

    warnings = check_training_domain(parameters, metadata)

    result = predict(
        model,
        feature_scaler,
        target_scaler,
        list(parameters.values()),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "inputs": parameters,
                    "predictions": result,
                    "training_domain_warnings": warnings,
                },
                indent=2,
            )
        )
        return

    print("IDM phase-transition surrogate prediction")
    print("-" * 48)

    for name, value in parameters.items():
        print(f"Input {name:<6} = {value:.8g}")

    print()

    if warnings:
        print("WARNING: input outside training domain")
        print("-" * 48)
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Training-domain check: all inputs within range")

    print()
    print("Predicted observables")
    print("-" * 48)
    print(f"Tcrit          = {result['Tcrit']:.8g}")
    print(f"Tnuc           = {result['Tnuc']:.8g}")
    print(f"alpha          = {result['alpha']:.8g}")
    print(f"beta/H         = {result['beta/H']:.8g}")


if __name__ == "__main__":
    main()
