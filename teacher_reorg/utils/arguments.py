import os
import yaml


def save_args(args, root_dir):
    args_file = os.path.join(root_dir, 'args.yaml')
    with open(args_file, 'w') as f:
        sorted_args = dict(sorted(vars(args).items()))
        yaml.dump(sorted_args, f, default_flow_style=False)
    print(f"Saving arguments to {args_file}")


def load_args(args, root_dir):
    args_file = os.path.join(root_dir, 'args.yaml')
    
    if not os.path.exists(args_file):
        print(f"No arguments file found at {args_file}, using default arguments")
        return args
    
    print(f"Loading arguments from {args_file}")
    with open(args_file, 'r') as f:
        loaded_args = yaml.safe_load(f)
    for key, value in loaded_args.items():
        setattr(args, key, value)
    return args
