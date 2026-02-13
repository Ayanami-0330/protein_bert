#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

BASELINE_EXPS = [
    "Baseline_ProteinBERT",
    "Ablation_RPSSM_110",
    "Exp15_ProteinBERT_PSSM310",
]

TARGET_EXPS = [
    "Exp16_ProteinBERT_PSSM710",
    "Exp17_ProteinBERT_PSSM1110",
    "Exp18_ProteinBERT_PSSM510",
]

METRIC_COLS = ["AUC", "AUPRC", "F1", "MCC", "Brier", "ECE"]


def _collect_result_files(features_dir: Path) -> List[Path]:
    candidates = sorted(features_dir.glob("exp*_results.csv"))
    return [p for p in candidates if p.name != "exp_summary.csv"]


def _load_and_merge_results(features_dir: Path) -> pd.DataFrame:
    files = _collect_result_files(features_dir)
    if not files:
        raise FileNotFoundError(f"No exp*_results.csv found in {features_dir}")

    parts = []
    for path in files:
        df = pd.read_csv(path)
        required = {"Exp", "Seed", *METRIC_COLS}
        if not required.issubset(set(df.columns)):
            continue
        parts.append(df[["Exp", "Seed", *METRIC_COLS, *([c for c in df.columns if c == "Threshold"]) ]])

    if not parts:
        raise RuntimeError("No valid experiment result tables found.")

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["Exp", "Seed"], keep="last").reset_index(drop=True)
    return merged


def _summarize(merged: pd.DataFrame) -> pd.DataFrame:
    summary = merged.groupby("Exp")[METRIC_COLS].agg(["mean", "std"])
    return summary.sort_values(("AUPRC", "mean"), ascending=False)


def _flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    return out


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def _get_metric(summary_flat: pd.DataFrame, exp: str, metric: str) -> float:
    row = summary_flat.loc[summary_flat["Exp"] == exp]
    if row.empty:
        raise KeyError(exp)
    return float(row.iloc[0][f"{metric}_mean"])


def _write_baseline_lock(features_dir: Path, summary_flat: pd.DataFrame) -> Path:
    out_path = features_dir / "baseline_lock.md"
    lines = [
        "# Baseline Lock",
        "",
        "固定后续对照基线（不扩 seed）：",
        "",
    ]
    for exp in BASELINE_EXPS:
        row = summary_flat.loc[summary_flat["Exp"] == exp]
        if row.empty:
            continue
        auc = _fmt(float(row.iloc[0]["AUC_mean"]))
        auprc = _fmt(float(row.iloc[0]["AUPRC_mean"]))
        mcc = _fmt(float(row.iloc[0]["MCC_mean"]))
        ece = _fmt(float(row.iloc[0]["ECE_mean"]))
        lines.append(f"- {exp}: AUC={auc}, AUPRC={auprc}, MCC={mcc}, ECE={ece}")

    lines.extend(
        [
            "",
            "说明：后续 Exp16/Exp17/Exp18 均相对 `Exp15_ProteinBERT_PSSM310` 判定是否纳入主线。",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _decision_rows(summary_flat: pd.DataFrame) -> List[Dict[str, str]]:
    rows = []
    if summary_flat.loc[summary_flat["Exp"] == "Exp15_ProteinBERT_PSSM310"].empty:
        return rows

    base_auprc = _get_metric(summary_flat, "Exp15_ProteinBERT_PSSM310", "AUPRC")
    base_mcc = _get_metric(summary_flat, "Exp15_ProteinBERT_PSSM310", "MCC")
    base_ece = _get_metric(summary_flat, "Exp15_ProteinBERT_PSSM310", "ECE")

    for exp in TARGET_EXPS:
        row = summary_flat.loc[summary_flat["Exp"] == exp]
        if row.empty:
            continue
        auprc = float(row.iloc[0]["AUPRC_mean"])
        mcc = float(row.iloc[0]["MCC_mean"])
        ece = float(row.iloc[0]["ECE_mean"])

        main_ok = (auprc >= base_auprc) and (mcc >= base_mcc)
        secondary_ok = (mcc > base_mcc) or (ece < base_ece)
        keep = main_ok and secondary_ok

        rows.append(
            {
                "Exp": exp,
                "AUPRC_mean": _fmt(auprc),
                "MCC_mean": _fmt(mcc),
                "ECE_mean": _fmt(ece),
                "main_rule": "pass" if main_ok else "fail",
                "secondary_rule": "pass" if secondary_ok else "fail",
                "decision": "KEEP" if keep else "DROP",
            }
        )
    return rows


def _write_decision_report(features_dir: Path, summary_flat: pd.DataFrame) -> Path:
    out_path = features_dir / "decision_report.md"
    lines = [
        "# Decision Report",
        "",
        "判定规则：",
        "- 主指标：相对 `Exp15_ProteinBERT_PSSM310`，`AUPRC mean` 与 `MCC mean` 同时不下降。",
        "- 次指标：`MCC` 与 `ECE` 至少一项改善。",
        "",
    ]
    rows = _decision_rows(summary_flat)
    if not rows:
        lines.append("当前缺少 Exp16/Exp17/Exp18 的结果，暂无新增特征纳入结论。")
    else:
        lines.append("| Exp | AUPRC_mean | MCC_mean | ECE_mean | 主指标 | 次指标 | 结论 |")
        lines.append("|---|---:|---:|---:|---|---|---|")
        for row in rows:
            lines.append(
                f"| {row['Exp']} | {row['AUPRC_mean']} | {row['MCC_mean']} | {row['ECE_mean']} | "
                f"{row['main_rule']} | {row['secondary_rule']} | {row['decision']} |"
            )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_comparison(features_dir: Path, summary_flat: pd.DataFrame) -> Path:
    out_path = features_dir / "comparison_onepage.md"
    interested = BASELINE_EXPS + TARGET_EXPS
    rows = summary_flat[summary_flat["Exp"].isin(interested)].copy()
    rows = rows.sort_values("AUPRC_mean", ascending=False)
    lines = [
        "# Baseline vs PSSM Variants",
        "",
        "| Exp | AUC(mean) | AUPRC(mean) | MCC(mean) | ECE(mean) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.Exp} | {_fmt(row.AUC_mean)} | {_fmt(row.AUPRC_mean)} | {_fmt(row.MCC_mean)} | {_fmt(row.ECE_mean)} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plan reports from experiment outputs.")
    parser.add_argument("--features-dir", required=True)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    merged = _load_and_merge_results(features_dir)
    summary = _summarize(merged)
    summary_flat = _flatten_summary(summary)

    exp_results = features_dir / "exp_results.csv"
    exp_summary = features_dir / "exp_summary.csv"
    exp_summary_flat = features_dir / "exp_summary_flat.csv"
    merged.to_csv(exp_results, index=False)
    summary.to_csv(exp_summary)
    summary_flat.to_csv(exp_summary_flat, index=False)

    baseline_lock = _write_baseline_lock(features_dir, summary_flat)
    decision_report = _write_decision_report(features_dir, summary_flat)
    onepage = _write_comparison(features_dir, summary_flat)

    print(f"Wrote: {exp_results}")
    print(f"Wrote: {exp_summary}")
    print(f"Wrote: {exp_summary_flat}")
    print(f"Wrote: {baseline_lock}")
    print(f"Wrote: {decision_report}")
    print(f"Wrote: {onepage}")


if __name__ == "__main__":
    main()
