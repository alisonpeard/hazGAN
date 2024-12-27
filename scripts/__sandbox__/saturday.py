# %%
# %load_ext autoreload
# %autoreload 2  
# %%
from environs import Env
from hazGAN.torch import load_data, WGANGP
from hazGAN.constants import SAMPLE_CONFIG

# %%
if __name__ == "__main__":
    env = Env()
    env.read_env(recurse=True)
    datadir = env.str("TRAINDIR")

    # Transforms dev
    train, valid, metadata = load_data(datadir, 64, "reflect", (18, 22), device='mps')
   
    model = WGANGP(SAMPLE_CONFIG)
    model.compile()
    # %%
    x = next(iter(train))
    label = x['label']
    label
    # %%

    # label = self.label_to_features(label)
    # Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)
    model = model.to('mps')
    model.fit(train, epochs=1)

# %%
