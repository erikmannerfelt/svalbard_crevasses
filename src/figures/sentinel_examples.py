"""Sentinel-2 image pairs beside the vote-fraction rasters, for demonstration."""

from __future__ import annotations

import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows

from src import settings

# Vote fractions below this are drawn transparent so the image stays readable;
# the color scale starts here so it does not imply values that are not drawn.
VOTE_DISPLAY_MIN = 0.05
# A dark halo keeps white annotations legible over snow.
HALO = (matplotlib.patheffects.withStroke(linewidth=1.5, foreground="0.15"),)
# Percentiles mapped to black and white when an example asks for a stretch.
STRETCH_PERCENTILES = (0.5, 99.9)


def read_example(
    example: settings.SentinelExample, date: str
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Read one image crop and the vote fraction of its year on the same grid.

    Returns
    -------
    rgb
        uint8 array of shape (rows, cols, 3).
    vote_fraction
        float32 array of shape (rows, cols), 0--1, nearest-neighbour resampled
        from the 50 m raster so its native pixels stay visible.
    extent
        (left, right, bottom, top) in metres of the image's UTM zone.
    """
    path = settings.SENTINEL_DIR / f"{example.file_stem}_{date}.tif"
    with rasterio.open(path) as image_file:
        window = rasterio.windows.from_bounds(*example.bounds, transform=image_file.transform)
        window = window.round_offsets().round_lengths()
        rgb = np.moveaxis(image_file.read(window=window), 0, -1)
        transform = image_file.window_transform(window)
        crs = image_file.crs

    vote_fraction = np.zeros(rgb.shape[:2], dtype=np.float32)
    vote_path = settings.RASTER_DIR / settings.VOTE_RASTER_TEMPLATE.format(year=int(date[:4]))
    with rasterio.open(vote_path) as vote_file:
        rasterio.warp.reproject(
            vote_file.read(1).astype(np.float32) / 255,
            vote_fraction,
            src_transform=vote_file.transform,
            src_crs=vote_file.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=rasterio.warp.Resampling.nearest,
        )
    left, bottom, right, top = rasterio.transform.array_bounds(*rgb.shape[:2], transform)
    return rgb, vote_fraction, (left, right, bottom, top)


def stretch_contrast(
    images: list[np.ndarray],
    extents: list[tuple[float, float, float, float]],
    exclude: tuple[float, float, float, float] | None,
) -> list[np.ndarray]:
    """Linearly stretch images with limits shared across both dates.

    Limits are shared across dates so their brightness stays comparable. The
    black point is common to all bands, but the white point is set per band:
    it lands on snow, so this also removes the export's slight purple cast
    without turning blue ice brown, which a fully per-band stretch does.
    Pixels inside `exclude` (left, bottom, right, top) do not set the limits.
    """
    samples = []
    for image, (left, right, bottom, top) in zip(images, extents, strict=True):
        keep = np.ones(image.shape[:2], dtype=bool)
        if exclude is not None:
            rows, cols = image.shape[:2]
            x = left + (np.arange(cols) + 0.5) * (right - left) / cols
            y = top - (np.arange(rows) + 0.5) * (top - bottom) / rows
            in_x = (x >= exclude[0]) & (x <= exclude[2])
            in_y = (y >= exclude[1]) & (y <= exclude[3])
            keep = ~(in_y[:, None] & in_x[None, :])
        samples.append(image[keep])
    pixels = np.concatenate(samples)
    low = np.percentile(pixels, STRETCH_PERCENTILES[0])
    high = np.percentile(pixels, STRETCH_PERCENTILES[1], axis=0)
    return [(np.clip((image.astype(np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8) for image in images]


def add_scale_bar(ax: plt.Axes, extent: tuple[float, float, float, float], length_m: float) -> None:
    """Draw a white scale bar in the lower left of a map panel."""
    left, right, bottom, top = extent
    x0 = left + 0.06 * (right - left)
    y0 = bottom + 0.07 * (top - bottom)
    ax.plot(
        [x0, x0 + length_m],
        [y0, y0],
        color="white",
        lw=2.5,
        solid_capstyle="butt",
        path_effects=(matplotlib.patheffects.withStroke(linewidth=3.5, foreground="0.15"),),
    )
    ax.text(
        x0 + length_m / 2,
        y0 + 0.035 * (top - bottom),
        f"{length_m / 1000:g} km",
        color="white",
        ha="center",
        va="bottom",
        fontsize=6.5,
        path_effects=HALO,
    )


def make_figure(output_path: object) -> None:
    """Render every configured example as a 2x2 block: images above, votes below."""
    fig = plt.figure(figsize=(7.0, 7.9))
    outer = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.06, left=0.01, right=0.99, top=0.96, bottom=0.1)

    image_artist = None
    for index, example in enumerate(settings.SENTINEL_EXAMPLES):
        block = outer[index // 2, index % 2].subgridspec(2, 2, hspace=0.03, wspace=0.03)
        crops = [read_example(example, date) for date in example.dates]
        images = [rgb for rgb, _, _ in crops]
        if example.stretch:
            images = stretch_contrast(images, [extent for _, _, extent in crops], example.stretch_exclude)
        for column, (date, rgb, (_, vote_fraction, extent)) in enumerate(
            zip(example.dates, images, crops, strict=True)
        ):
            image_ax = fig.add_subplot(block[0, column])
            vote_ax = fig.add_subplot(block[1, column])

            image_ax.imshow(rgb, extent=extent, interpolation="none")
            # A dimmed greyscale backdrop keeps orientation without competing with the colors.
            vote_ax.imshow(rgb.mean(axis=2), extent=extent, cmap="gray", vmin=-120, vmax=300, interpolation="none")
            image_artist = vote_ax.imshow(
                np.ma.masked_less(vote_fraction, VOTE_DISPLAY_MIN),
                extent=extent,
                cmap="YlOrRd",
                vmin=VOTE_DISPLAY_MIN,
                vmax=1,
                alpha=0.9,
                interpolation="none",
            )
            for ax in (image_ax, vote_ax):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
            image_ax.text(
                0.03,
                0.97,
                date,
                transform=image_ax.transAxes,
                va="top",
                fontsize=7,
                color="white",
                fontweight="bold",
                path_effects=HALO,
            )
            if column == 0:
                add_scale_bar(image_ax, extent, 1000)

        title_ax = fig.add_subplot(block[0, :], frame_on=False)
        title_ax.set_xticks([])
        title_ax.set_yticks([])
        title_ax.set_title(f"({'abcd'[index]}) {example.name} ({example.cause})", fontsize=8, loc="left", pad=3)

    colorbar_ax = fig.add_axes((0.3, 0.055, 0.4, 0.012))
    colorbar = fig.colorbar(image_artist, cax=colorbar_ax, orientation="horizontal")
    ticks = [VOTE_DISPLAY_MIN, 0.5, 1]
    colorbar.set_ticks(ticks=ticks, labels=[f"{tick:.0%}" for tick in ticks], fontsize=7)
    colorbar.set_label("Model vote fraction", fontsize=7, labelpad=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def run() -> None:
    """Render the Sentinel-2 comparison figure."""
    make_figure(settings.FIGURE_DIR / "sentinel-examples.png")


if __name__ == "__main__":
    run()
