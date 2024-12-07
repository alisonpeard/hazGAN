# check for missing month/year combinations in the data
import os
from environs import Env
import glob

if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    indir = env.str("ERA5DIR")
    indir = os.path.join(indir, "original")
    print("Checking for missing files in", indir)
    files = glob.glob(os.path.join(indir, f"bangladesh_*.nc"))
    files = [os.path.basename(file) for file in files]
    files = [file.split(".")[0] for file in files]
    files = set(files)

    years = range(1950, 2023)
    months = range(1, 13)

    missing = []
    for year in years:
        for month in months:
            if f"bangladesh_{year}" not in files:
                missing.append(f"{year}")
    print(len(missing), "missing")
    print(missing)
# %%

