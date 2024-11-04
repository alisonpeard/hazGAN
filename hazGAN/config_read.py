# %%
import os
import sys
import yaml



def read_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'local_config.yaml')
    try:
        with open(config_path, 'r') as f:
            try:
                localconfig = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
                sys.exit(1)
            return localconfig
    except FileNotFoundError as e:
        print(e)
        print("You must set local_config.yaml in the main folder. Copy local_config-example.yaml and adjust appropriately.")
        sys.exit(1)
# %%