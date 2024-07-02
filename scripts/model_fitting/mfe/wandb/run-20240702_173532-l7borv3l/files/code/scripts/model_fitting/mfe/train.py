import wandb

# defaults = dict(seed=42, filter_size=3)
# wandb.init(entity="xxx", allow_val_change=True, project="xxx",config=defaults)

wandb.init(allow_val_change=True, project="mfe")
print('\n\nHyperparams:')
print(wandb.config)
print('\n\n')
log = 2*wandb.config["filter_size"] - 0.01*wandb.config["seed"]
wandb.log({"output":log})
wandb.finish()