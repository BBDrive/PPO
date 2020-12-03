import gym
import torch
import numpy as np
import multiprocessing as mp

from collections import namedtuple

Transition = namedtuple('Transition', ('observation', 'value', 'action', 'logproba', 'mask', 'reward'))

Get_Enough_Batch = mp.Value('i', 0)  # 0还不够 1batch足够


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


class EnvWorker(mp.Process):
    def __init__(self, remote, env, queue, lock, seed):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = env
        self.queue = queue
        self.lock = lock

        # seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def run(self):
        while True:
            command, policy = self.remote.recv()
            if command == 'collect':
                while Get_Enough_Batch.value == 0:
                    episode = Episode()
                    observation = self.env.reset()
                    while Get_Enough_Batch.value == 0:
                        with torch.no_grad():
                            observation_tensor = torch.from_numpy(observation).float().unsqueeze(0)
                            action_mean, action_logstd, value = policy(observation_tensor)
                            action, logproba = policy.select_action(action_mean, action_logstd)
                            action = action.data.cpu().numpy()[0]
                            logproba = logproba.data.cpu().numpy()[0]
                            value = value.data.cpu().numpy()[0][0]

                        new_observation, reward, done, _ = self.env.step(action)
                        mask = 0 if done else 1
                        episode.push(observation, value, action, logproba, mask, reward)
                        if done:
                            with self.lock:
                                self.queue.put(episode)
                            break
                        observation = new_observation

            elif command == 'close':
                self.remote.close()
                self.env.close()
                break
            elif command == 'get_spaces':
                obs_shape = np.prod(self.env.observation_space.shape)
                acs_shape = np.prod(self.env.action_space.shape)
                self.remote.send((obs_shape, acs_shape))
            else:
                raise NotImplementedError()


class MemorySampler(object):
    def __init__(self, args):
        self.env_name = args.env_name
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.device = args.device

        self.queue = mp.Queue()
        self.lock = mp.Lock()
        self.envs = [make_env(self.env_name, self.seed + i) for i in range(self.num_workers)]

        # Pipe方法返回(conn1, conn2)代表一个管道的两个端，
        # Pipe方法有duplex参数，如果duplex参数为True(默认值)，那么这个管道是全双工模式，也就是说conn1和conn2均可收发。
        # duplex为False，conn1只负责接受消息，conn2只负责发送消息。
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in self.envs])

        self.workers = [EnvWorker(remote, env, self.queue, self.lock, args.seed)
                        for (remote, env) in zip(self.work_remotes, self.envs)]
        for worker in self.workers:
            # 如果某个子线程的daemon属性为False，主线程结束时会检测该子线程是否结束，如果该子线程还在运行，则主线程会等待它完成后再退出
            # 如果某个子线程的daemon属性为True，主线程运行结束时不对这个子线程进行检查而直接退出，同时所有daemon值为True的子线程将随主线程一起结束，而不论是否运行完成。
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()

    def sample(self, policy):
        policy.to('cpu')
        memory = Memory()
        Get_Enough_Batch.value = 0
        for remote in self.remotes:
            remote.send(('collect', policy))

        while len(memory) < self.batch_size:
            episode = self.queue.get(True)
            memory.push(episode)

        Get_Enough_Batch.value = 1

        while self.queue.qsize() > 0:
            self.queue.get()

        policy.to(self.device)
        return memory

    @property
    def get_space(self):
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        return observation_space, action_space

    def close(self):
        Get_Enough_Batch.value = 1
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
