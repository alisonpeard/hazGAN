# %%
# %load_ext autoreload
# %autoreload 2  
# %%
from torch import mps
from environs import Env
from hazGAN.torch import load_data, WGANGP, MemoryLogger
from hazGAN.constants import SAMPLE_CONFIG


# %%
if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    # %% Transforms dev
    train, valid, metadata = load_data(datadir, 128, "reflect", (18, 22), device='mps')

    # %% MPS memory management
    print(f"Recommended max: {mps.recommended_max_memory() / 1e9:.2f} GB")
    print(f"Driver allocated: {mps.driver_allocated_memory() / 1e9:.2f} GB")
    print(f"Current allocated: {mps.current_allocated_memory() / 1e9:.2f} GB")

    # %% final check on data
    print("Final data statistics:\n----------------------")
    print("Train sample mean:", next(iter(train))['uniform'].mean())
    print("Train sample std:", next(iter(train))['uniform'].std())
    print("Validation sample mean:", next(iter(valid))['uniform'].mean())
    print("Validation sample std:", next(iter(valid))['uniform'].std())
    # 
    mps.empty_cache()
    model = WGANGP(SAMPLE_CONFIG)
    model.compile()

    # %%
    model.fit(train, epochs=1, callbacks=[MemoryLogger(100)])
    # %%
