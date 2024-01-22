import yaml

def read_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config