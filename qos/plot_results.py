"""python -m qos.plot_results results/main.csv --out results/main.png
Bar chart if the CSV has one setting ("base"); line plot per policy if it is a sweep ("lam=2", "lam=3", ...)."""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--out", default=None); ap.add_argument("--metric", default="qos")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    err = "qos_ci95" if a.metric == "qos" else None
    if df["setting"].nunique() == 1:
        ax.bar(df["policy"], df[a.metric], yerr=df[err] if err else None, capsize=3)
        ax.set_ylabel(a.metric)
    else:
        xs = df["setting"].str.split("=").str[1]
        df["x"] = pd.to_numeric(xs, errors="coerce") if pd.to_numeric(xs, errors="coerce").notna().all() else xs
        for name, g in df.groupby("policy"):
            ax.errorbar(g["x"], g[a.metric], yerr=g[err] if err else None, marker="o", capsize=2, label=name)
        ax.set_xlabel(df["setting"].iloc[0].split("=")[0]); ax.set_ylabel(a.metric); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); fig.tight_layout()
    out = a.out or a.csv.replace(".csv", ".png")
    fig.savefig(out, dpi=200); print("saved", out)


if __name__ == "__main__":
    main()
