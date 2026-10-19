from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns


INPUT = Path("data/final/ghcnh_conus_precip_2020_2022_compact.parquet")
OUT_DIR = Path("analysis/ghcnh_precip_2020_2022_stats/yearly_facetgrid_distribution")


def sample_for_plot(lf: pl.LazyFrame, per_group: int = 350_000) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in [2020, 2021, 2022]:
        year_lf = lf.filter(pl.col("year") == year)
        with_zero = (
            year_lf.select(["year", "precip_mm", "log1p_precip_mm"])
            .collect()
            .to_pandas()
            .sample(n=per_group, random_state=year)
        )
        with_zero["subset"] = "All observations"
        frames.append(with_zero)

        positive_df = (
            year_lf.filter(pl.col("precip_mm") > 0)
            .select(["year", "precip_mm", "log1p_precip_mm"])
            .collect()
            .to_pandas()
        )
        positive_df = positive_df.sample(
            n=min(per_group, len(positive_df)), random_state=year + 10
        )
        positive_df["subset"] = "Positive only"
        frames.append(positive_df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lf = (
        pl.scan_parquet(INPUT)
        .select(["report_time_utc", "precip_mm"])
        .with_columns(
            pl.col("report_time_utc").dt.year().alias("year"),
            pl.col("precip_mm").log1p().alias("log1p_precip_mm"),
        )
    )

    summary = (
        lf.group_by("year")
        .agg(
            pl.len().alias("rows"),
            (pl.col("precip_mm") == 0).sum().alias("zero_rows"),
            (pl.col("precip_mm") > 0).sum().alias("positive_rows"),
            pl.col("precip_mm").mean().alias("mean_mm"),
            pl.col("precip_mm").quantile(0.95).alias("p95_mm"),
            pl.col("precip_mm").quantile(0.99).alias("p99_mm"),
            pl.col("precip_mm").quantile(0.999).alias("p999_mm"),
            pl.col("precip_mm").max().alias("max_mm"),
        )
        .with_columns(
            (pl.col("zero_rows") / pl.col("rows")).alias("zero_share"),
            (pl.col("positive_rows") / pl.col("rows")).alias("positive_share"),
        )
        .sort("year")
        .collect()
        .to_pandas()
    )
    summary.to_csv(OUT_DIR / "yearly_precip_summary.csv", index=False)

    positive_summary = (
        lf.filter(pl.col("precip_mm") > 0)
        .group_by("year")
        .agg(
            pl.len().alias("positive_rows"),
            pl.col("precip_mm").mean().alias("positive_mean_mm"),
            pl.col("precip_mm").quantile(0.001).alias("positive_p001_mm"),
            pl.col("precip_mm").quantile(0.01).alias("positive_p01_mm"),
            pl.col("precip_mm").quantile(0.5).alias("positive_p50_mm"),
            pl.col("precip_mm").quantile(0.95).alias("positive_p95_mm"),
            pl.col("precip_mm").quantile(0.99).alias("positive_p99_mm"),
            pl.col("precip_mm").quantile(0.999).alias("positive_p999_mm"),
            pl.col("precip_mm").max().alias("positive_max_mm"),
        )
        .sort("year")
        .collect()
        .to_pandas()
    )
    positive_summary.to_csv(OUT_DIR / "yearly_positive_precip_summary.csv", index=False)

    plot_df = sample_for_plot(lf)
    plot_df["year"] = plot_df["year"].astype(str)
    plot_df["subset"] = pd.Categorical(
        plot_df["subset"], ["All observations", "Positive only"], ordered=True
    )
    plot_df.to_parquet(OUT_DIR / "yearly_facetgrid_plot_sample.parquet", index=False)

    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.titlesize": 17,
            "axes.linewidth": 1.3,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    g = sns.FacetGrid(
        plot_df,
        row="subset",
        col="year",
        sharex=True,
        sharey=False,
        height=3.2,
        aspect=1.25,
        margin_titles=True,
        despine=True,
    )
    g.map_dataframe(
        sns.histplot,
        x="log1p_precip_mm",
        bins=80,
        stat="density",
        common_norm=False,
        color="#2563EB",
        edgecolor=None,
        alpha=0.72,
    )
    g.set_axis_labels("log(1 + precipitation mm h$^{-1}$)", "Density")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ax in g.axes.flat:
        ax.grid(True, axis="y", color="0.86", linewidth=0.9)
        ax.grid(False, axis="x")

    g.figure.subplots_adjust(top=0.94, wspace=0.18, hspace=0.30)
    png = OUT_DIR / "yearly_precip_distribution_facetgrid_log1p.png"
    svg = OUT_DIR / "yearly_precip_distribution_facetgrid_log1p.svg"
    g.figure.savefig(png, dpi=300, bbox_inches="tight")
    g.figure.savefig(svg, bbox_inches="tight")
    plt.close(g.figure)

    print(summary.to_string(index=False))
    print()
    print(positive_summary.to_string(index=False))
    print()
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
