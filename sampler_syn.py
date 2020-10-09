import gym
import torch
import multiprocessing as mp
import highway_env

from subproc_vec_env import SubprocVecEnv
from collections import namedtuple

Transition = namedtuple('Transition', ('observation', 'value', 'action', 'logproba', 'mask', 'reward'))


def make_env(env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    return env


class Episode(object):
    def __init__(self):
        self.episode = []

    def push(self, *args):
        self.episode.append(Transition(*args))

    def __len__(self):
        return len(self.episode)


class Memory(object):
    def __init__(self):
        self.memory = []
        self.num_episode = 0

    def push(self, epi: Episode):
        self.memory += epi.episode
        self.num_episode += 1

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class MemorySampler(object):
    def __init__(self, args):
        """

        :param env_name: 某个环境的名字
        :param num_workers: 进程数
        """

        self.env_name = args.env_name
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.device = args.device
        self.batch_size = args.batch_size

        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(self.env_name, self.seed) for _ in range(self.num_workers)],
                                  queue=self.queue)

    # sample一个环境中的batch_size条trajectories
    def sample(self, policy):

        memory = Memory()
        Episode_dict = {i: Episode() for i in range(2*self.num_workers)}

        for id in range(2*self.num_workers):
            self.queue.put(id)
        observations, batch_ids = self.envs.reset()

        while True:
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).float().to(device=self.device)
                action_means, action_logstds, values = policy(observations_tensor)
                actions, logprobas = policy.select_action(action_means, action_logstds)
                actions = actions.data.cpu().numpy()
                logprobas = logprobas.data.cpu().numpy()
                values = values.data.cpu().numpy()

            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)

            for observation, value, action, logproba, done, reward, batch_id in zip(
                    observations, values, actions, logprobas, dones, rewards, batch_ids):
                mask = 0 if done else 1
                Episode_dict[batch_id].push(observation, value, action, logproba, mask, reward)

                if done:
                    memory.push(Episode_dict[batch_id])
                    del Episode_dict[batch_id]

                    if len(memory) >= self.batch_size:
                        # clear queue
                        while self.queue.qsize() > 0:
                            self.queue.get()
                        return memory

                    id += 1
                    self.queue.put(id)
                    Episode_dict.update({id: Episode()})

            observations, batch_ids = new_observations, new_batch_ids

    @property
    def get_space(self):
        return self.envs.observation_space, self.envs.action_space
