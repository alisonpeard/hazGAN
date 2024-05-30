# check for missing month/year combinations in the data
import os
import glob

indir = os.path.join(os.path.expandvars("$HOME"), "data", "new_data", "era5")

files = glob.glob(os.path.join(indir, f"*bangladesh*.nc"))
files = [os.path.basename(file) for file in files]
files = [file.split(".")[0] for file in files]
files = set(files)

years = range(1950, 2023)
months = range(1, 13)

missing = []
for year in years:
    for month in months:
        if f"bangladesh_{year}_{str(month).zfill(2)}" not in files:
            missing.append(f"{year}_{str(month).zfill(2)}")
print(missing)
# %%

