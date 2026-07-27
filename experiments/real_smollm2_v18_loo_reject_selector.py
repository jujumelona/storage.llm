from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "observable_selector_v17", HERE / "real_smollm2_v17_observable_instance_selector.py"
)
v17 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(v17)

v16 = v17.v16
ROOT = Path("out/real_smollm2_v18_loo_reject_selector")
ROOT.mkdir(parents=True, exist_ok=True)
v16.ROOT = ROOT

ORIGINAL_FIT = v16.v14.fit_ridge_gcv
FIT_CALLS: list[dict[str, Any]] = []


def loo_reject_fit(x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> dict[str, Any]:
    fit = ORIGINAL_FIT(x, y, sample_weight)

    feature_mean = fit["feature_mean"]
    feature_std = fit["feature_std"]
    z = (x - feature_mean) / feature_std
    design = np.concatenate([z, np.ones((len(z), 1), dtype=np.float64)], axis=1)

    sqrt_weight = np.sqrt(sample_weight)[:, None]
    xw = design * sqrt_weight
    yw = y * sqrt_weight[:, 0]
    u, singular_values, _ = np.linalg.svd(xw, full_matrices=False)
    lam = float(fit["lambda"])
    leverage = (u.square() * (singular_values.square() / (singular_values.square() + lam))[None, :]).sum(axis=1)

    weighted_fitted = xw @ fit["coefficient"]
    weighted_residual = yw - weighted_fitted
    loo_weighted_prediction = yw - weighted_residual / np.maximum(1.0 - leverage, 1e-8)
    loo_prediction = loo_weighted_prediction / np.maximum(sqrt_weight[:, 0], 1e-30)

    # Asymmetric reject rule: Instruct is the safe general default.  Only
    # positive evidence may switch to the SQL specialist.
    candidates = np.unique(np.concatenate([[0.0], loo_prediction[loo_prediction >= 0.0]]))
    best = None
    for threshold in candidates:
        predicted = np.where(loo_prediction > threshold, 1.0, -1.0)
        weighted_error = float(np.sum(sample_weight * (predicted != y)) / np.sum(sample_weight))
        sql_fraction = float(np.mean(predicted > 0.0))
        candidate = (weighted_error, -float(threshold), sql_fraction, float(threshold))
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    threshold = best[3]

    # Compile the threshold into the intercept so all downstream code and the
    # serialized checkpoint retain the exact decision rule without a new API.
    fit["coefficient"] = fit["coefficient"].copy()
    fit["coefficient"][-1] -= threshold
    shifted_scores = design @ fit["coefficient"]
    fit["calibration_accuracy"] = float(np.mean(np.sign(shifted_scores) == y))
    fit["loo_decision_threshold_raw"] = threshold
    fit["loo_weighted_route_error"] = best[0]
    fit["loo_sql_fraction"] = best[2]
    FIT_CALLS.append(
        {
            "n": len(y),
            "threshold": threshold,
            "loo_weighted_route_error": best[0],
            "loo_sql_fraction": best[2],
        }
    )
    return fit


v16.v14.fit_ridge_gcv = loo_reject_fit


if __name__ == "__main__":
    v16.main()
    results_path = ROOT / "RESULTS.json"
    result = json.loads(results_path.read_text(encoding="utf-8"))
    result["selector_observation"] = {
        "uses_all_inference_observable_text": True,
        "mcq_answer_choices_included": True,
        "target_or_correct_label_included": False,
    }
    result["asymmetric_reject_rule"] = {
        "default_parent": "instruct",
        "switch_condition": "raw ridge score > nonnegative LOO-calibrated threshold",
        "threshold_selected_without_heldout_labels": True,
        "final_fit": FIT_CALLS[-1] if FIT_CALLS else None,
        "all_fit_calls": FIT_CALLS,
    }
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    report_path = ROOT / "REPORT.md"
    final_fit = FIT_CALLS[-1] if FIT_CALLS else {}
    report_path.write_text(
        report_path.read_text(encoding="utf-8")
        + "\n\nThe SQL route uses a nonnegative reject threshold selected from calibration-only ridge leave-one-out predictions."
        + f" Final raw threshold: **{final_fit.get('threshold')}**."
        + " Uncertain inputs remain on the Instruct parent.\n",
        encoding="utf-8",
    )
