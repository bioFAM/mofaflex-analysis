import argparse
import json
import os

import pandas as pd


def _load_table(path: str) -> pd.DataFrame:
    path = os.path.expanduser(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".json"):
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    raise ValueError("Input must be .csv or .json")


def _normalize_name_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Embedding" in out.columns:
        return out
    unnamed = [c for c in out.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        return out.rename(columns={unnamed[0]: "Embedding"})
    if out.columns.size > 0:
        return out.rename(columns={out.columns[0]: "Embedding"})
    raise ValueError("Could not infer embedding-name column.")


def _extract_metric(df: pd.DataFrame, metric: str, alias: str) -> pd.DataFrame:
    out = _normalize_name_column(df)
    if metric not in out.columns:
        raise KeyError(f"Missing metric column '{metric}'.")
    out = out[["Embedding", metric]].copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out[out[metric].notna()].copy()
    out = out[out["Embedding"] != "Metric Type"].copy()
    out = out.rename(columns={metric: alias})
    return out


def main(args):
    left = _extract_metric(_load_table(args.left_input), args.left_metric, args.left_alias)
    right = _extract_metric(_load_table(args.right_input), args.right_metric, args.right_alias)

    merged = left.merge(right, on="Embedding", how="outer")
    merged[f"{args.left_alias} rank"] = merged[args.left_alias].rank(ascending=False, method="min")
    merged[f"{args.right_alias} rank"] = merged[args.right_alias].rank(ascending=False, method="min")
    merged = merged.sort_values(args.left_alias, ascending=False, na_position="last").reset_index(drop=True)

    out_csv = os.path.expanduser(args.out_csv)
    out_json = os.path.expanduser(args.out_json)
    if os.path.dirname(out_csv):
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.dirname(out_json):
        os.makedirs(os.path.dirname(out_json), exist_ok=True)

    merged.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(merged.to_dict(orient="records"), f, indent=2)

    print(merged)
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--left-input", required=True)
    p.add_argument("--left-metric", required=True)
    p.add_argument("--left-alias", default=None)
    p.add_argument("--right-input", required=True)
    p.add_argument("--right-metric", required=True)
    p.add_argument("--right-alias", default=None)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()
    if args.left_alias is None:
        args.left_alias = args.left_metric
    if args.right_alias is None:
        args.right_alias = args.right_metric
    main(args)
