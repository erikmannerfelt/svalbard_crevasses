"""Small plotting functions for inspecting model envelopes and slopes."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_regional_model_envelope(annual_path: Path, group: str = "all", ax=None):
    """Plot annual regional coverage with the model ensemble envelope."""
    annual = pd.read_feather(annual_path).query("group == @group")
    yearly = annual.groupby("year").coverage_km2.agg(
        median="median", low=lambda values: values.quantile(0.025), high=lambda values: values.quantile(0.975)
    )
    ax = ax or plt.subplots()[1]
    ax.fill_between(yearly.index, yearly.low, yearly.high, alpha=0.2)
    ax.plot(yearly.index, yearly.median, marker="o")
    ax.set(xlabel="Year", ylabel="Positive crevasse area (km²)", title=f"{group}: model ensemble")
    return ax


def plot_glacier_slope_distribution(trends_path: Path, ax=None):
    """Plot glacier median slopes, split by the preserved surge flag."""
    trends = pd.read_feather(trends_path)
    ax = ax or plt.subplots()[1]
    for surging, label in [(False, "Non-surging"), (True, "Surging")]:
        ax.hist(trends.loc[trends.surging == surging, "slope_km2_yr"], bins=40, alpha=0.55, label=label)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Trend (km²/year)", ylabel="Glacier count")
    ax.legend()
    return ax
