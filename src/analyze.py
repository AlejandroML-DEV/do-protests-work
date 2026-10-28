"""Reproducible NAVCO 1.3 campaign outcome analysis."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "NAVCO_1_3_List.xlsx"
OUTPUT = ROOT / "outputs"
OUTCOMES = ["SUCCESS", "LIMITED", "FAILURE", "ONGOING"]
FLAGS = OUTCOMES + ["VIOL", "NONVIOL", "FSELFDET", "REGCHANGE", "SECESSION", "OTHER"]
OBJECTIVES = ["FSELFDET", "REGCHANGE", "SECESSION", "OTHER"]
YEAR_CENTER = 1960.0


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logit(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Fit a logit by Newton scoring and return country-cluster sandwich SEs."""
    beta = np.zeros(x.shape[1], dtype=float)
    for _ in range(100):
        probability = sigmoid(x @ beta)
        weight = np.clip(probability * (1 - probability), 1e-9, None)
        score = x.T @ (y - probability)
        information = x.T @ (weight[:, None] * x)
        step = np.linalg.solve(information, score)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    else:
        raise RuntimeError("Logistic regression did not converge after 100 iterations")

    probability = sigmoid(x @ beta)
    weight = np.clip(probability * (1 - probability), 1e-9, None)
    bread = np.linalg.inv(x.T @ (weight[:, None] * x))
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        cluster_score = x[mask].T @ (y[mask] - probability[mask])
        meat += np.outer(cluster_score, cluster_score)
    cluster_count = len(np.unique(groups))
    n, k = x.shape
    correction = (cluster_count / (cluster_count - 1)) * ((n - 1) / (n - k))
    covariance = bread @ meat @ bread * correction
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return {"beta": beta, "covariance": covariance, "se": se, "probability": probability,
            "clusters": cluster_count, "n": n, "iterations": _ + 1}


def normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def model_table(name: str, fit: dict, terms: list[str]) -> list[dict]:
    result = []
    for index, term in enumerate(terms):
        beta, se = fit["beta"][index], fit["se"][index]
        z = beta / se if se > 0 else math.nan
        result.append({
            "model": name, "term": term, "coefficient": beta, "cluster_se": se,
            "odds_ratio": math.exp(float(np.clip(beta, -30, 30))),
            "ci_low_or": math.exp(float(np.clip(beta - 1.96 * se, -30, 30))),
            "ci_high_or": math.exp(float(np.clip(beta + 1.96 * se, -30, 30))),
            "p_value_normal": normal_p(z), "n": fit["n"], "country_clusters": fit["clusters"],
        })
    return result


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def svg_chart(path: Path, title: str, subtitle: str, body: str, width: int = 960, height: int = 560) -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f7f3"/>
<text x="64" y="58" font-family="Arial,sans-serif" font-size="29" font-weight="700" fill="#1e2a2d">{title}</text>
<text x="64" y="88" font-family="Arial,sans-serif" font-size="15" fill="#59676a">{subtitle}</text>
{body}</svg>'''
    path.write_text(svg, encoding="utf-8")


def bar_chart(path: Path, title: str, subtitle: str, labels: list[str], values: list[float],
              value_format: str = "percent", colors: list[str] | None = None) -> None:
    left, right, top, bottom = 100, 890, 145, 430
    colors = colors or ["#267c78"] * len(labels)
    maximum = max(values) if values else 1
    ceiling = 100 if value_format == "percent" else max(1, maximum * 1.15)
    pieces = []
    for tick in range(0, 101, 20) if value_format == "percent" else range(0, 101, 20):
        y = bottom - (tick / 100) * (bottom - top)
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dce2df"/>')
        pieces.append(f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#637174">{tick}%</text>')
    slot = (right - left) / len(labels)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        h = value / ceiling * (bottom - top)
        x = left + i * slot + slot * 0.22
        y = bottom - h
        shown = f"{value:.1f}%" if value_format == "percent" else f"{value:,.0f}"
        pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot*.56:.1f}" height="{h:.1f}" rx="8" fill="{color}"/>')
        pieces.append(f'<text x="{x+slot*.28:.1f}" y="{y-12:.1f}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="#1e2a2d">{shown}</text>')
        pieces.append(f'<text x="{x+slot*.28:.1f}" y="{bottom+28}" text-anchor="middle" font-family="Arial" font-size="14" fill="#334246">{label}</text>')
    pieces.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#879494"/>')
    svg_chart(path, title, subtitle, "\n".join(pieces))


def line_chart(path: Path, title: str, subtitle: str, rows: pd.DataFrame) -> None:
    left, right, top, bottom = 100, 890, 145, 430
    pieces = []
    for tick in range(0, 101, 20):
        y = bottom - tick / 100 * (bottom - top)
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dce2df"/>')
        pieces.append(f'<text x="{right+14}" y="{y+5:.1f}" text-anchor="start" font-family="Arial" font-size="13" fill="#267c78">{tick}%</text>')
    decades = rows["decade"].to_numpy()
    rates = rows["success_rate"].to_numpy() * 100
    counts = rows["campaigns"].to_numpy()
    x_points = left + (decades - decades.min()) / max(1, decades.max() - decades.min()) * (right - left)
    y_points = bottom - rates / 100 * (bottom - top)
    max_count = max(counts)
    slot = (right - left) / max(1, len(decades))
    for x, count in zip(x_points, counts):
        h = count / max_count * (bottom - top) * 0.78
        pieces.append(f'<rect x="{x-slot*.27:.1f}" y="{bottom-h:.1f}" width="{slot*.54:.1f}" height="{h:.1f}" rx="4" fill="#c8d7d3"/>')
    for tick in range(0, 101, 20):
        y = bottom - tick / 100 * (bottom - top) * 0.78
        pieces.append(f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#637174">{round(max_count*tick/100)}</text>')
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_points, y_points))
    pieces.append(f'<polyline points="{points}" fill="none" stroke="#267c78" stroke-width="4"/>')
    for x, y, year, rate, count in zip(x_points, y_points, decades, rates, counts):
        pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#267c78"/>')
        pieces.append(f'<text x="{x:.1f}" y="{bottom+27}" text-anchor="middle" font-family="Arial" font-size="12" fill="#334246">{year}s</text>')
        pieces.append(f'<text x="{x:.1f}" y="{y-13:.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#1e2a2d">{rate:.0f}% · n={count}</text>')
    pieces.append(f'<text x="{left}" y="{top-14}" font-family="Arial" font-size="12" fill="#637174">Campaign count (bars)</text>')
    pieces.append(f'<text x="{right}" y="{top-14}" text-anchor="end" font-family="Arial" font-size="12" fill="#267c78">Full-success rate (line)</text>')
    svg_chart(path, title, subtitle, "\n".join(pieces))


def objective_chart(path: Path, summary: pd.DataFrame) -> None:
    labels = ["Self-determination", "Regime change", "Secession", "Other"]
    variables = ["FSELFDET", "REGCHANGE", "SECESSION", "OTHER"]
    left, right, top, bottom = 110, 890, 145, 430
    pieces = []
    for tick in range(0, 101, 20):
        y = bottom - tick / 100 * (bottom - top)
        pieces.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dce2df"/>')
        pieces.append(f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#637174">{tick}%</text>')
    group_width = (right - left) / len(variables)
    for i, (variable, label) in enumerate(zip(variables, labels)):
        group = summary[summary["objective"].eq(variable)].set_index("category")
        for j, category in enumerate(["No", "Yes"]):
            row = group.loc[category]
            x = left + i * group_width + group_width * (0.22 + j * 0.30)
            h = row["success_rate"] * (bottom - top)
            y = bottom - h
            color = "#c8d7d3" if category == "No" else "#267c78"
            pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{group_width*.23:.1f}" height="{h:.1f}" rx="6" fill="{color}"/>')
            pieces.append(f'<text x="{x+group_width*.115:.1f}" y="{y-9:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#334246">{row["success_rate"]:.0%}</text>')
        cx = left + i * group_width + group_width / 2
        pieces.append(f'<text x="{cx:.1f}" y="{bottom+28}" text-anchor="middle" font-family="Arial" font-size="12" fill="#334246">{label}</text>')
    pieces.append(f'<rect x="{left}" y="{top-22}" width="12" height="12" fill="#c8d7d3"/><text x="{left+18}" y="{top-12}" font-family="Arial" font-size="12" fill="#59676a">No</text>')
    pieces.append(f'<rect x="{left+65}" y="{top-22}" width="12" height="12" fill="#267c78"/><text x="{left+83}" y="{top-12}" font-family="Arial" font-size="12" fill="#59676a">Yes</text>')
    svg_chart(path, "Full success by stated campaign objective", "Completed campaigns · binary objective indicators compared descriptively; sample sizes are in the CSV", "\n".join(pieces))


def forest_chart(path: Path, predicted: dict) -> None:
    width, height = 960, 500
    left, right = 230, 850
    top, bottom = 205, 390
    min_x, max_x = 0.1, 0.8
    pieces = []
    for tick in np.arange(0.1, 0.81, 0.1):
        x = left + (tick - min_x) / (max_x - min_x) * (right - left)
        pieces.append(f'<line x1="{x:.1f}" y1="{top-18}" x2="{x:.1f}" y2="{bottom+15}" stroke="#dce2df"/>')
        pieces.append(f'<text x="{x:.1f}" y="{bottom+42}" text-anchor="middle" font-family="Arial" font-size="13" fill="#637174">{tick:.0%}</text>')
    for i, (label, key) in enumerate([("Violent", "violent"), ("Nonviolent", "nonviolent")]):
        y = top + i * 115
        lo, p, hi = predicted[key]
        xlo = left + (lo - min_x) / (max_x - min_x) * (right - left)
        xp = left + (p - min_x) / (max_x - min_x) * (right - left)
        xhi = left + (hi - min_x) / (max_x - min_x) * (right - left)
        pieces.append(f'<text x="{left-20}" y="{y+5}" text-anchor="end" font-family="Arial" font-size="18" font-weight="700" fill="#1e2a2d">{label}</text>')
        pieces.append(f'<line x1="{xlo:.1f}" y1="{y}" x2="{xhi:.1f}" y2="{y}" stroke="#267c78" stroke-width="5"/>')
        pieces.append(f'<line x1="{xlo:.1f}" y1="{y-10}" x2="{xlo:.1f}" y2="{y+10}" stroke="#267c78" stroke-width="3"/>')
        pieces.append(f'<line x1="{xhi:.1f}" y1="{y-10}" x2="{xhi:.1f}" y2="{y+10}" stroke="#267c78" stroke-width="3"/>')
        pieces.append(f'<circle cx="{xp:.1f}" cy="{y}" r="9" fill="#d56b45" stroke="#f7f7f3" stroke-width="3"/>')
        pieces.append(f'<text x="{xp:.1f}" y="{y-20}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700" fill="#1e2a2d">{p:.1%} [{lo:.1%}, {hi:.1%}]</text>')
    pieces.append(f'<line x1="{left}" y1="{bottom+15}" x2="{right}" y2="{bottom+15}" stroke="#879494"/>')
    svg_chart(path, "Adjusted predicted probability of full success", "Model 2 · covariates held to each completed campaign's observed values; 95% clustered delta-method CI", "\n".join(pieces), width, height)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(SOURCE)
    data = raw.copy()
    data["DURATION"] = data["EYEAR"] - data["BYEAR"] + 1
    data["DECADE"] = (data["BYEAR"] // 10) * 10

    missing = raw.isna().sum().rename("missing_count").to_frame()
    missing["missing_pct"] = missing["missing_count"] / len(raw) * 100
    missing.to_csv(OUTPUT / "missingness.csv")
    profile = pd.DataFrame({
        "variable": raw.columns,
        "dtype": [str(raw[column].dtype) for column in raw.columns],
        "non_missing": [int(raw[column].notna().sum()) for column in raw.columns],
        "missing": [int(raw[column].isna().sum()) for column in raw.columns],
        "distinct_non_missing": [int(raw[column].nunique(dropna=True)) for column in raw.columns],
        "sample_values": [json.dumps(raw[column].dropna().astype(str).unique()[:5].tolist(), ensure_ascii=False)
                          for column in raw.columns],
    })
    profile.to_csv(OUTPUT / "column_profile.csv", index=False)
    quality = {
        "rows": int(len(raw)), "columns": int(raw.shape[1]),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_NAVCOID_rows": int(raw["NAVCOID"].duplicated(keep=False).sum()),
        "repeated_NAVCOID_values": int(raw.loc[raw["NAVCOID"].duplicated(keep=False), "NAVCOID"].nunique()),
        "unique_NAVCOID": int(raw["NAVCOID"].nunique()),
        "locations": int(raw["LOCATION"].nunique()),
        "binary_flag_values_valid": bool(all(raw[c].dropna().isin([0, 1]).all() for c in FLAGS)),
        "invalid_year_order_rows": int((raw["EYEAR"] < raw["BYEAR"]).sum()),
        "start_year_min": int(raw["BYEAR"].min()), "start_year_max": int(raw["BYEAR"].max()),
        "end_year_min": int(raw["EYEAR"].min()), "end_year_max": int(raw["EYEAR"].max()),
        "ongoing_rows": int(raw["ONGOING"].eq(1).sum()),
        "undefined_outcome_rows": int(raw[OUTCOMES].sum(axis=1).eq(0).sum()),
        "multi_status_rows": int(raw[OUTCOMES].sum(axis=1).gt(1).sum()),
        "ongoing_also_failure_rows": int((raw["ONGOING"].eq(1) & raw["FAILURE"].eq(1)).sum()),
        "violent_nonviolent_exclusive": bool((raw["VIOL"] + raw["NONVIOL"]).eq(1).all()),
        "outcome_flag_counts": {c: int(raw[c].sum()) for c in OUTCOMES},
    }
    (OUTPUT / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    status_sum = data[OUTCOMES].sum(axis=1)
    completed = data.loc[data["ONGOING"].eq(0) & status_sum.gt(0)].copy()
    # Full success is the requested outcome; limited success remains in the comparison group.
    completed["SUCCESS"] = completed["SUCCESS"].astype(int)
    outcome_dist = raw[OUTCOMES].sum().rename_axis("outcome").reset_index(name="campaigns")
    outcome_dist.to_csv(OUTPUT / "outcome_distribution.csv", index=False)
    bar_chart(OUTPUT / "outcome_distribution.svg", "Campaign outcomes in NAVCO 1.3",
              "All 622 campaigns · outcome indicators can overlap for ongoing campaigns",
              ["Success", "Limited", "Failure", "Ongoing"],
              [raw[c].sum() / len(raw) * 100 for c in OUTCOMES],
              colors=["#267c78", "#76a8a0", "#d56b45", "#aa6252"])

    grouped = completed.groupby("NONVIOL")["SUCCESS"].agg(["count", "sum", "mean"]).reindex([0, 1])
    grouped["ci_low"], grouped["ci_high"] = zip(*[
        wilson_interval(int(row["sum"]), int(row["count"])) for _, row in grouped.iterrows()
    ])
    grouped.rename(index={0: "violent", 1: "nonviolent"}).to_csv(OUTPUT / "strategy_success_rates.csv")
    n0, n1 = int(grouped.loc[0, "count"]), int(grouped.loc[1, "count"])
    p0, p1 = float(grouped.loc[0, "mean"]), float(grouped.loc[1, "mean"])
    difference = p1 - p0
    diff_se = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    pooled = (int(grouped.loc[0, "sum"]) + int(grouped.loc[1, "sum"])) / (n0 + n1)
    z = difference / math.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    comparison = {"violent_n": n0, "violent_successes": int(grouped.loc[0, "sum"]), "violent_rate": p0,
                  "violent_ci95": list(wilson_interval(int(grouped.loc[0, "sum"]), n0)),
                  "nonviolent_n": n1, "nonviolent_successes": int(grouped.loc[1, "sum"]), "nonviolent_rate": p1,
                  "nonviolent_ci95": list(wilson_interval(int(grouped.loc[1, "sum"]), n1)),
                  "difference_nonviolent_minus_violent": difference,
                  "difference_ci95_wald": [difference - 1.96 * diff_se, difference + 1.96 * diff_se],
                  "two_proportion_z": z, "two_sided_p_value": normal_p(z),
                  "test": "pooled two-proportion z-test; normal approximation"}
    (OUTPUT / "descriptive_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    bar_chart(OUTPUT / "success_by_strategy.svg", "Full success by campaign strategy",
              "Completed campaigns only · 95% Wilson intervals are in the CSV table",
              [f"Violent · n={n0}", f"Nonviolent · n={n1}"], [100 * p0, 100 * p1],
              colors=["#d56b45", "#267c78"])

    temporal = completed.groupby("DECADE").agg(campaigns=("NAVCOID", "size"), successes=("SUCCESS", "sum"))
    temporal["success_rate"] = temporal["successes"] / temporal["campaigns"]
    temporal.reset_index(names="decade").to_csv(OUTPUT / "temporal_summary.csv", index=False)
    line_chart(OUTPUT / "success_by_decade.svg", "Campaign volume and success over time",
               "Completed campaigns by start decade · labels show full-success rate and campaign count",
               temporal.reset_index(names="decade"))

    objective_rows = []
    for variable in OBJECTIVES:
        for value, label in [(1, "Yes"), (0, "No")]:
            sample = completed.loc[completed[variable].eq(value), "SUCCESS"]
            objective_rows.append({"objective": variable, "category": label, "campaigns": int(sample.size),
                                   "full_successes": int(sample.sum()), "success_rate": float(sample.mean()) if len(sample) else math.nan})
    objective_summary = pd.DataFrame(objective_rows)
    objective_summary.to_csv(OUTPUT / "objective_success_rates.csv", index=False)
    objective_chart(OUTPUT / "success_by_objective.svg", objective_summary)

    completed["duration_group"] = pd.qcut(completed["DURATION"], 5, duplicates="drop")
    duration_summary = completed.groupby("duration_group", observed=True).agg(
        campaigns=("SUCCESS", "size"), success_rate=("SUCCESS", "mean"),
        duration_median=("DURATION", "median"), duration_min=("DURATION", "min"), duration_max=("DURATION", "max")
    ).reset_index()
    duration_summary["duration_group"] = duration_summary["duration_group"].astype(str)
    duration_summary.to_csv(OUTPUT / "duration_summary.csv", index=False)
    bar_chart(OUTPUT / "success_by_duration.svg", "Success rate across duration groups",
              "Completed campaigns · quantile groups of campaign duration in years; association is not assumed linear",
              [f"{int(r.duration_median)}y median" for r in duration_summary.itertuples()],
              [r.success_rate * 100 for r in duration_summary.itertuples()],
              colors=["#267c78", "#397f84", "#55928d", "#76a8a0", "#d56b45"][:len(duration_summary)])

    # The four objective indicators are exhaustive and mutually exclusive in these rows;
    # omit OTHER as the reference category to avoid perfect collinearity with the intercept.
    objective_controls = ["FSELFDET", "REGCHANGE", "SECESSION"]
    model_data = completed.dropna(subset=OBJECTIVES).copy()
    model_data["YEAR_DECADES"] = (model_data["BYEAR"] - YEAR_CENTER) / 10
    model_data["LOG_DURATION"] = np.log1p(model_data["DURATION"])
    covariates = ["INTERCEPT", "NONVIOL", "YEAR_DECADES", *objective_controls]
    model_data["INTERCEPT"] = 1.0
    term_names = ["Intercept", "NONVIOL", "BYEAR per decade", "FSELFDET vs OTHER", "REGCHANGE vs OTHER", "SECESSION vs OTHER"]
    matrices: dict[str, tuple[np.ndarray, list[str]]] = {}
    base_terms = ["INTERCEPT", "NONVIOL"]
    matrices["Model 1 · strategy only"] = (model_data[base_terms].to_numpy(float), ["Intercept", "NONVIOL"])
    adjusted = covariates + ["LOG_DURATION"]
    matrices["Model 2 · adjusted"] = (model_data[adjusted].to_numpy(float), term_names + ["log(1 + duration)"])
    without_duration = [c for c in covariates]
    matrices["Model 3 · no duration"] = (model_data[without_duration].to_numpy(float), term_names)
    fits = {}
    table = []
    y = model_data["SUCCESS"].to_numpy(float)
    clusters = model_data["LOCATION"].to_numpy()
    for name, (matrix, terms) in matrices.items():
        fit = fit_logit(matrix, y, clusters)
        fits[name] = fit
        table.extend(model_table(name, fit, terms))
    pd.DataFrame(table).to_csv(OUTPUT / "model_coefficients.csv", index=False)

    # Average predictions for each campaign after toggling only NONVIOL; delta-method CIs.
    fit = fits["Model 2 · adjusted"]
    matrix, model_terms = matrices["Model 2 · adjusted"]
    term_index = model_terms.index("NONVIOL")
    prediction_results = {}
    prediction_gradients = {}
    for label, exposure in [("violent", 0.0), ("nonviolent", 1.0)]:
        counterfactual = matrix.copy()
        counterfactual[:, term_index] = exposure
        probs = sigmoid(counterfactual @ fit["beta"])
        gradient = np.mean((probs * (1 - probs))[:, None] * counterfactual, axis=0)
        standard_error = math.sqrt(max(0, float(gradient @ fit["covariance"] @ gradient)))
        estimate = float(probs.mean())
        prediction_results[label] = [max(0, estimate - 1.96 * standard_error), estimate,
                                      min(1, estimate + 1.96 * standard_error)]
        prediction_gradients[label] = gradient
    adjusted_difference = prediction_results["nonviolent"][1] - prediction_results["violent"][1]
    difference_gradient = prediction_gradients["nonviolent"] - prediction_gradients["violent"]
    difference_se = math.sqrt(max(0, float(difference_gradient @ fit["covariance"] @ difference_gradient)))
    prediction_results["difference"] = [max(-1, adjusted_difference - 1.96 * difference_se),
                                         adjusted_difference,
                                         min(1, adjusted_difference + 1.96 * difference_se)]
    (OUTPUT / "adjusted_predictions.json").write_text(json.dumps(prediction_results, indent=2), encoding="utf-8")
    forest_chart(OUTPUT / "adjusted_predicted_success.svg", prediction_results)

    report = {
        "dataset": {"source_file": SOURCE.name, "observations": len(raw), "years_start": [int(raw.BYEAR.min()), int(raw.BYEAR.max())],
                    "countries": int(raw.LOCATION.nunique()), "analysis_observations": len(completed),
                    "excluded_ongoing": quality["ongoing_rows"], "excluded_undefined_outcome": quality["undefined_outcome_rows"],
                    "model_observations": len(model_data), "excluded_missing_objective_for_models": len(completed) - len(model_data)},
        "comparison": comparison,
        "models": {name: {"n": fit["n"], "country_clusters": fit["clusters"],
                           "nonviol_beta": float(fit["beta"][1]), "nonviol_se": float(fit["se"][1]),
                           "nonviol_or": float(math.exp(fit["beta"][1])),
                           "nonviol_ci95_or": [float(math.exp(fit["beta"][1] - 1.96 * fit["se"][1])),
                                                float(math.exp(fit["beta"][1] + 1.96 * fit["se"][1]))],
                           "nonviol_p": float(normal_p(fit["beta"][1] / fit["se"][1]))}
                   for name, fit in fits.items()},
        "adjusted_predictions": prediction_results,
    }
    (OUTPUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
