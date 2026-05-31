"""
Single-axis comprehensive figure
一张图、一对XY轴，叠加所有信息：
  - SAR + RGB 归一化直方图（density=True，可与KDE叠加）
  - SAR + RGB KDE曲线
  - 11px / 31px 核大小竖线
  - 覆盖率标注
  - 峰值标注
  - 阴影覆盖区域
  - 统计信息文字框

用法：
python code/mask_scale_analysis.py     --sar_csv ./scale_analysis_results/stats_sar.csv     --rgb_csv ./scale_analysis_results/stats_rgb.csv     --output ./results_v3
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

COLOR_SAR   = "#2775B6"
COLOR_RGB   = "#E8603C"
COLOR_SHORT = "#27AE60"
COLOR_LONG  = "#8E44AD"


def make_single_axis_fig(df_sar, df_rgb, output_dir):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    # ── 数据准备 ────────────────────────────────────────────
    def get_minor(df, clip=99):
        d = df["minor_axis_px"].dropna()
        d = d[d > 0]
        return d[d <= np.percentile(d, clip)]

    ds = get_minor(df_sar)
    dr = get_minor(df_rgb)
    x_max = max(np.percentile(ds, 99), np.percentile(dr, 99))

    # ── 归一化直方图（density，可与KDE叠在同一y轴）──────────
    ax.hist(ds, bins=60, density=True, color=COLOR_SAR, alpha=0.25,
            edgecolor="white", linewidth=0.3, label="_nolegend_")
    ax.hist(dr, bins=60, density=True, color=COLOR_RGB, alpha=0.25,
            edgecolor="white", linewidth=0.3, label="_nolegend_")

    # ── KDE 曲线 ────────────────────────────────────────────
    for label, d, color in [("SAR", ds, COLOR_SAR), ("Optical (RGB)", dr, COLOR_RGB)]:
        kde = stats.gaussian_kde(d, bw_method=0.15)
        x = np.linspace(0, x_max, 800)
        y = kde(x)
        ax.plot(x, y, color=color, lw=2.8, zorder=4,
                label=f"{label}  (n={len(d):,}, median={np.median(d):.1f} px)")
        ax.fill_between(x, y, alpha=0.10, color=color, zorder=3)

        # 峰值箭头标注
        peak_x = x[np.argmax(y)]
        peak_y = y.max()
        offset_x = 4 if label == "SAR" else 3
        offset_y = -peak_y * 0.18 if label == "SAR" else peak_y * 0.12
        ax.annotate(
            f"{label} peak ≈ {peak_x:.1f} px",
            xy=(peak_x, peak_y),
            xytext=(peak_x + offset_x, peak_y + offset_y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.6),
            fontsize=10, color=color, fontweight="bold"
        )

    # ── 阴影覆盖区域 ─────────────────────────────────────────
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.35
    ax.axvspan(0,  11, alpha=0.08, color=COLOR_SHORT, zorder=1,
               label="Short-range kernel coverage (0–11 px)")
    ax.axvspan(11, 31, alpha=0.06, color=COLOR_LONG,  zorder=1,
               label="Long-range kernel coverage (11–31 px)")

    # ── 核大小竖线 + 覆盖率标注 ──────────────────────────────
    sar_full  = df_sar["minor_axis_px"].dropna(); sar_full = sar_full[sar_full > 0]
    rgb_full  = df_rgb["minor_axis_px"].dropna(); rgb_full = rgb_full[rgb_full > 0]

    for k, kc, kname in [(11, COLOR_SHORT, "Short-range"), (31, COLOR_LONG, "Long-range")]:
        ax.axvline(x=k, color=kc, lw=2.2, ls="--", zorder=5,
                   label=f"DASC {kname} kernel ({k} px)")

        c_sar = (sar_full <= k).sum() / len(sar_full) * 100
        c_rgb = (rgb_full <= k).sum() / len(rgb_full) * 100

        # 覆盖率文字：竖线顶部
        ax.text(k + 0.3, ymax * 0.97,
                f"SAR {c_sar:.1f}%\nRGB {c_rgb:.1f}%",
                color=kc, fontsize=9.5, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=kc,
                          alpha=0.85, lw=1.2))

    # ── 中位数虚线 ───────────────────────────────────────────
    for d, color, label in [(sar_full, COLOR_SAR, "SAR"), (rgb_full, COLOR_RGB, "RGB")]:
        med = np.median(d)
        ax.axvline(med, color=color, lw=1.2, ls=":", alpha=0.7,
                   label=f"{label} median = {med:.1f} px")

    # ── 右上角统计信息文本框 ─────────────────────────────────
    def pct(d, k): return (d <= k).sum() / len(d) * 100
    info = (
        f"Statistical Summary\n"
        f"{'─'*28}\n"
        f"SAR   n = {len(sar_full):,}\n"
        f"  Median : {np.median(sar_full):.1f} px\n"
        f"  P95    : {np.percentile(sar_full,95):.1f} px\n"
        f"  ≤11 px : {pct(sar_full,11):.1f}%\n"
        f"  ≤31 px : {pct(sar_full,31):.1f}%\n"
        f"{'─'*28}\n"
        f"Optical  n = {len(rgb_full):,}\n"
        f"  Median : {np.median(rgb_full):.1f} px\n"
        f"  P95    : {np.percentile(rgb_full,95):.1f} px\n"
        f"  ≤11 px : {pct(rgb_full,11):.1f}%\n"
        f"  ≤31 px : {pct(rgb_full,31):.1f}%"
    )
    ax.text(0.99, 0.97, info,
            transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec="gray", alpha=0.90, lw=1.0))

    # ── 坐标轴 ───────────────────────────────────────────────
    ax.set_xlim(0, x_max)
    ax.set_ylim(0)
    ax.set_xlabel("Minor Axis Length — Wavelength Direction (pixels)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Internal Wave Spatial Scale Distribution vs. DASC Kernel Sizes\n"
        "Statistical justification of kernel size selection across SAR and Optical modalities",
        fontsize=13, fontweight="bold", pad=10
    )
    ax.grid(alpha=0.22, linestyle="--")
    ax.legend(fontsize=9.5, framealpha=0.90, loc="upper center",
              bbox_to_anchor=(0.42, 0.98), ncol=2)

    # ── 保存 ─────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) / "fig_single_axis.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"✅ 保存: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sar_csv", required=True)
    parser.add_argument("--rgb_csv", required=True)
    parser.add_argument("--output",  default="./results_v3")
    args = parser.parse_args()

    df_sar = pd.read_csv(args.sar_csv)
    df_rgb = pd.read_csv(args.rgb_csv)
    make_single_axis_fig(df_sar, df_rgb, args.output)


if __name__ == "__main__":
    main()