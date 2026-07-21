import geopandas as gpd
import rasterio as rio
import rasterio.features
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
def nmad(arr, axis=None):
    median = np.nanmedian(arr, axis=axis, keepdims=True)

    return 1.4826 * np.nanmedian(np.abs(arr - median), axis=axis) 

def surge_overlaps_interval(term_value, onset_value, min_year, max_year) -> bool:
    def _tokens(value):
        if pd.isna(value):
            return []
        return [part.strip() for part in str(value).split(";") if part.strip()]

    onset_tokens = _tokens(onset_value)
    term_tokens = _tokens(term_value)
    for idx, onset_token in enumerate(onset_tokens):
        try:
            onset_year = int(onset_token)
        except ValueError:
            continue
        if onset_year > max_year:
            continue
        term_token = term_tokens[idx] if idx < len(term_tokens) else None
        if term_token is None:
            term_year = np.inf
        else:
            term_lower = term_token.lower()
            if term_lower in {"", "n/a", "not observed", "ongoing"}:
                term_year = np.inf
            else:
                try:
                    term_year = int(term_token)
                except ValueError:
                    term_year = np.inf
        if onset_year <= max_year and term_year > min_year:
            return True
    return False

def read_outlines():
    start_year = 2013
    end_year = 2025
    
    field_overrides = {
        "RGI2000-v7.0-G-07-00741": {"term_type": 1},  # Scheelebreen became tidewater during the study period; treat it as tidewater throughout.
        "RGI2000-v7.0-G-07-00780": {"S_Term": 2017}, # Approximate for Vasilievbreen
        "RGI2000-v7.0-G-07-01312": {"glac_name": "Lilliehöökbreen"},
        "RGI2000-v7.0-G-07-01379": {"S_Onset": "2015", "S_Term": "Ongoing", "glac_name": "Austfonna Basin-7"},
        "RGI2000-v7.0-G-07-01383": {"glac_name": "Storisstraumen"},
        "RGI2000-v7.0-G-07-01385": {"glac_name": "Austfonna Basin-2"},
        "RGI2000-v7.0-G-07-01381": {"glac_name": "Austfonna Basin-5"},
        "RGI2000-v7.0-G-07-01384": {"glac_name": "Bråsvellbreen"},
        "RGI2000-v7.0-G-07-00560": {"S_Onset": "2012", "S_Term": "Ongoing"},  # dH/dt observation of Tinkarpbreen
        "RGI2000-v7.0-G-07-00888": {"S_Onset": "2016", "S_Term": "2019"},  # Sveitsarfonna mirroring Penckbreen
        "RGI2000-v7.0-G-07-00899": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},  # Zawadzkibreen mirroring Nathorstbreen
        "RGI2000-v7.0-G-07-00902": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},  # Polakkbreen mirroring Nathorstbreen
        "RGI2000-v7.0-G-07-00912": {"S_Onset": "1890; 2008", "S_Term": "1890; 2017"},  # Dobrowolskibreen mirroring Nathorstbreen
    }
    # data = gpd.read_file("/home/erikmann/Projects/UiO/phd_thesis/data/RGI_V7_Surge_Database.zip")
    outlines = gpd.read_file("zip:///home/erikmann/Projects/UiO/phd_thesis/data/RGI_V7_Surge_Database.zip/RGI_V7_Surge_Database/RGI2000-v7.0-G-07_svalbard_jan_mayen_Surge_Database.shp").to_crs(32633)
    outlines["geometry"] = outlines["geometry"].buffer(0)

    for rgi_id, overrides in field_overrides.items():
        idx = outlines[outlines["rgi_id"] == rgi_id].index
        for key, value in overrides.items():
            outlines.loc[idx, key] = value

    outlines["surging"] = [surge_overlaps_interval(term_value, onset_value, start_year, end_year) for term_value, onset_value in zip(outlines["S_Term"], outlines["S_Onset"])]

    return outlines

def read_and_sample_outlines():

    cache_path = Path("cache/outlines.arrow")

    if not cache_path.is_file():

        outlines = read_outlines()
        outlines["id"] = np.arange(1, outlines.shape[0] + 1)

        outlines_rst = np.empty((0,))
        glacier_mask = outlines_rst.copy()
        pixel_area = 0

        sampled = {}
        for filepath in sorted(Path("/home/erikmann/Projects/UiO/phd_thesis/projects/foundation_crevasses").glob("*.tif")):
            if "crevasses_" not in filepath.stem:
                continue
            year = int(filepath.stem.replace("crevasses_", "")) 
            with rio.open(filepath) as raster:

                pixel_area = raster.res[0] * raster.res[1]

                if outlines_rst.shape[0] == 0:
                    outlines_rst = rasterio.features.rasterize(
                        outlines[["geometry", "id"]].values.tolist(),
                        (raster.height, raster.width),
                        transform=raster.transform,
                    )
                    glacier_mask = outlines_rst > 0
                    outlines_rst = outlines_rst[glacier_mask]

                crev = raster.read(1, masked=True).filled(0)[glacier_mask]

                sampled[str(year)] = crev
            
        sampled = pd.DataFrame(sampled)
        sampled["id"] = outlines_rst

        sampled = sampled.groupby("id").sum() * pixel_area / 1e6

        outlines = pd.merge(outlines, sampled, left_on="id", right_index=True)

        cache_path.parent.mkdir(exist_ok=True, parents=True)
        outlines.to_feather(cache_path)

    outlines = gpd.read_feather(cache_path)

    rename = {}
    for col in outlines.columns:
        if str(col).isnumeric():
            rename[col] = int(col)

    return outlines.rename(rename)



    

def main():

    outlines = read_and_sample_outlines()

    years = list(map(str, range(2017, 2025)))

    outlines = outlines[outlines["area_km2"] > 7]

    outlines["crev_area_std"] = 100 * np.std(outlines[years], axis=1) / outlines["area_km2"]
    outlines["crev_area_lsq"] =  outlines[years].apply(lambda vs: np.polyfit(vs.index.astype(int), vs, deg=1)[0], axis="columns")

    outlines["crev_area_lsq_percent"] = 100 * outlines["crev_area_lsq"] / outlines["area_km2"]

    # outlines[

    outlines.to_feather("cache/outlines_ext.arrow")
    # print(outlines["crev_area_lsq"])
    # return

    print(outlines.query("~surging").sort_values("crev_area_std").iloc[-5:][["rgi_id", "glac_name", "area_km2", *years]].round(2))

    plt.plot(outlines.query("surging")[years].sum(axis="rows"))
    plt.show()

    ax = None

    col = "crev_area_lsq"
    for i, query in enumerate(["surging", "~surging"]):
        ax = plt.subplot(2, 1, i + 1, sharex=ax)
        subset = outlines.query(query)
        plt.hist(subset[col], bins=50)

        if i == 1:
            plt.yscale("log")


    plt.show()



    
            

    

if __name__ == "__main__":
    main()
