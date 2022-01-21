import numpy as np
import torch
import yaml

from nrm import NRM


def main():
    with open('config/gail_config.yaml')as f:
        config = yaml.full_load(f)
    device = "cuda:0" if torch.cuda.is_available() and config['cuda'] else "cpu"
    config['device'] = device

    seed_num = config['env']['seed']
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    torch.set_num_threads(1)
    nrm = NRM(config)
    nrm.run()


if __name__ == "__main__":
    main()
