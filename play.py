import time
import yaml
import env_maker
import torch


def play(save_data_path, config_path):
    with open(config_path)as f:
        config = yaml.full_load(f)
    env = env_maker.make_play_env(config)
    data = torch.load(save_data_path)

    agent = data['actor']
    obs_filter = data['obs_filt']

    obs = env.reset()
    obs_filter.eval()
    while True:
        env.render()
        time.sleep(0.001)
        obs = obs_filter(obs)
        action, _, _ = agent.act(obs)
        obs, _, _, _ = env.step(action)


if __name__ == "__main__":
    play("R:\RL\SavedModel\mountain_car_16_done\mountain_car_16_done-990.pt"
         , "R:\RL\SavedModel\mountain_car_16_done\\archive_mountain_car_16_done_config.yaml")
