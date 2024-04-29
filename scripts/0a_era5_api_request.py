"""
Use ERA5 CDS API to request data year-by-year --run on cluster.

Uses conda env general.

Is it also possible to pre-process?

API installation instructions here: https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+macOS
"""

# %%
import cdsapi
import os
import numpy as np

datadir = os.path.join(
    "/Users", "alison", "Documents", "DPhil", "data", "era5", "new_data"
)
os.makedirs(datadir, exist_ok=True)

years = np.arange(1950, 2023)
months = np.arange(1, 13)

c = cdsapi.Client(timeout=600, quiet=False, debug=True)
for year in years:
    for month in months:
        if not os.path.exists(
            os.path.join(datadir, f"bangladesh_{year}_{str(month).zfill(2)}.nc")
        ):
            if not os.path.exists(
                os.path.join(datadir, f"bangladesh_{year}_{month}.nc")
            ):
                print(f"Requesting {year}-{month}")
                c.retrieve(
                    f"reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": [
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                            "mean_sea_level_pressure",
                        ],
                        "year": str(year),
                        "month": str(month).zfill(2),
                        "day": [
                            "01",
                            "02",
                            "03",
                            "04",
                            "05",
                            "06",
                            "07",
                            "08",
                            "09",
                            "10",
                            "11",
                            "12",
                            "13",
                            "14",
                            "15",
                            "16",
                            "17",
                            "18",
                            "19",
                            "20",
                            "21",
                            "22",
                            "23",
                            "24",
                            "25",
                            "26",
                            "27",
                            "28",
                            "29",
                            "30",
                            "31",
                        ],
                        "time": [
                            "00:00",
                            "01:00",
                            "02:00",
                            "03:00",
                            "04:00",
                            "05:00",
                            "06:00",
                            "07:00",
                            "08:00",
                            "09:00",
                            "10:00",
                            "11:00",
                            "12:00",
                            "13:00",
                            "14:00",
                            "15:00",
                            "16:00",
                            "17:00",
                            "18:00",
                            "19:00",
                            "20:00",
                            "21:00",
                            "22:00",
                            "23:00",
                        ],
                        "area": [
                            25,
                            80,
                            10,
                            95,
                        ],
                        "format": "netcdf",
                    },
                    os.path.join(
                        datadir, f"bangladesh_{year}_{str(month).zfill(2)}.nc"
                    ),
                )
# %%
