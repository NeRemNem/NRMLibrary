import os.path
from collections import defaultdict
from collections import deque
import torch
import env_maker
from algos import create_algorithm
from models import create_agent, create_intrinsic_model
from reward_value_factory import RewardValueFactory
from storage import Memory
from utils.log_manager import LogManager
from utils.normalizer import ObsFilter


class NRM:
    def __init__(self, config: dict):
        self.config = config
        self.main_algo = {}
        self.intrinsic_algo = {}
        self.episode_rewards = deque(maxlen=10)
        self.intrinsic_rewards = defaultdict(list)
        self.step = 0

        self.log_manager = LogManager(config)

        self.obs_filter = ObsFilter()
        self.env = env_maker.make_vec_envs(config)
        self.memory = Memory(self)
        # region model Setting
        self.agent = create_agent(self.env, config)
        self.intrinsic_models = create_intrinsic_model(self.env, config)
        # endregion

        self.main_algo, self.intrinsic_algo = create_algorithm(config)
        self.reward_value_factory = RewardValueFactory(self.intrinsic_models, self.memory, config)

        if len(config['reward'].keys()) < 1:
            raise
        if (len(config['reward'].keys()) > 1 and 'extrinsic' in config['reward']) \
                or (len(config['reward'].keys()) == 1 and 'extrinsic' not in config['reward']):
            self.use_intrinsic = True
        else:
            self.use_intrinsic = False

        self.losses = []

        if not os.path.isdir('SavedModel/' + config['run_id']):
            os.makedirs('SavedModel/' + config['run_id'])

    def run(self):

        total_step = self.config['max_step'] // self.config['env']['num_process'] // self.config['buffer_size']
        obs = self._warm_up()

        print("Start")
        # WORK FLOW
        for step in range(total_step):
            self.step = step
            obs = self._sampling(obs)
            self._before_update()
            self._update()
            self._after_update()

        print("DONE")

    def _warm_up(self):
        obs = self.env.reset()
        self.obs_filter.train()
        self.obs_filter(obs)

        for _ in range(128 * 50):
            action = torch.randint(0, self.env.action_space.n, size=(self.config['env']['num_process'], 1))
            obs, _, _, _ = self.env.step(action)
            self.obs_filter(obs)

        obs = self.env.reset()
        obs = self.obs_filter(obs)
        self.memory.ready(obs)
        return obs

    def _sampling(self, obs):
        for i in range(self.config['buffer_size']):
            # if self.step > 10:
            #     self.env.render()
            action, values, log_prob = self.agent.act(obs)
            obs, reward, done, infos = self.env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            obs = self.obs_filter(obs)
            self.memory.push(state=obs, action=action, log_prob=log_prob, mask=masks)
            self.reward_value_factory.push_extrinsic_reward(reward)
            self.reward_value_factory.push_values(values)
        self.reward_value_factory.push_values(self.agent.get_value(obs))
        return obs

    def _before_update(self):
        self.obs_filter.eval()

        self.reward_value_factory.make_early_reward()
        if self.use_intrinsic:
            if self.step < 10:
                # warm up
                for _ in range(10):
                    for name, algo in self.intrinsic_algo.items():
                        name, loss = algo.update(self.memory, self.reward_value_factory, self.intrinsic_models[name])
                        self.log_manager.add_sub_algo_loss(name, loss)
            else:
                for name, algo in self.intrinsic_algo.items():
                    name, loss = algo.update(self.memory, self.reward_value_factory, self.intrinsic_models[name])
                    self.log_manager.add_sub_algo_loss(name, loss)

        self.reward_value_factory.make_lazy_reward()
        self.log_manager.add_reward(self.reward_value_factory.get_rewards_mean())

    def _update(self):
        for k in self.main_algo.keys():
            loss = self.main_algo[k].update(self.memory, self.reward_value_factory, self.agent)
            self.log_manager.add_main_algo_loss(loss)

    def _after_update(self):
        self.obs_filter.train()
        self.memory.after_update()
        self._print_log()
        self._save_model()

    def _print_log(self):
        self.log_manager.print_log(self.step)

    def _save_model(self):
        if self.step % self.config['save_interval'] == 0 and self.step > 0:
            path = os.path.join("SavedModel", f"{self.config['run_id']}-{self.step}.pt")
            torch.save({
                'actor': self.agent
                , 'obs_filt': self.obs_filter
            }, path)
