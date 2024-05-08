import json
import os

config_path = os.environ["PLATFORM_CONFIG_PATH"]

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        dataset_path = list(config["dataset_map"].values())[0]
        model_path = list(config["model_map"].values())[0]
        dataset_name = list(config["dataset_map"].keys())[0]
        model_name = list(config["model_map"].keys())[0]

    pass
else:
    raise ValueError("jeeves config file not exists!")

print(dataset_path)

# export PLATFORM_CONFIG_PATH=/home/jeeves/openmatch/platform_config.json

# cpm_d_2b_embedding=$(python -c "print('hello')")


