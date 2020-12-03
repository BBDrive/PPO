import os
import time
import torch
import numpy as np
import torch.optim as opt

from sampler_asyn import MemorySampler
from model import ActorCritic
from tensorboardX import SummaryWriter

asyn = True
if asyn:
    from sampler_asyn import MemorySampler
else:
    from sampler_syn import MemorySampler


class args(object):
    env_name = 'HalfCheetah-v2'
    num_workers = 16  # 进程数
    seed = 1234  # 随机种子
    num_episode = 2  # 总回合
    batch_size = 2048  # 训练的数据量
    gamma = 0.995
    lamda = 0.97
    log_num_episode = 1  # 记录间隔
    num_epoch = 10  # ppo更新倍数
    minibatch_size = 256
    clip = 0.2
    loss_coeff_value = 0.5
    loss_coeff_entropy = 0.01
    lr = 3e-4
    EPS = 1e-10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tricks
    schedule_adam = 'linear'
    schedule_clip = 'linear'
    layer_norm = True
    state_norm = False  # 还未完善
    advantage_norm = True
    lossvalue_norm = True


def main(args):
    current_dir = os.path.abspath('.')
    exp_dir = current_dir + '/results/exp/'
    model_dir = current_dir + '/results/model/'
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(exp_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sampler = MemorySampler(args)
    num_inputs, num_actions = sampler.get_space

    network = ActorCritic(num_inputs, num_actions, layer_norm=args.layer_norm).to(args.device)
    optimizer = opt.Adam(network.parameters(), lr=args.lr)

    clip_now = args.clip

    for i_episode in range(args.num_episode):
        # step1: perform current policy to collect trajectories
        # this is an on-policy method!
        memory = sampler.sample(network)

        # step2: extract variables from trajectories
        batch = memory.sample()
        batch_size = len(memory)

        rewards = torch.Tensor(batch.reward)
        values = torch.Tensor(batch.value)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(batch.action)
        observations = torch.Tensor(batch.observation)
        oldlogproba = torch.Tensor(batch.logproba)

        returns = torch.Tensor(batch_size)
        deltas = torch.Tensor(batch_size)
        advantages = torch.Tensor(batch_size)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(batch_size)):
            returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + args.gamma * args.lamda * prev_advantage * masks[i]

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]
        if args.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + args.EPS)

        observations = observations.to(args.device)
        actions = actions.to(args.device)
        oldlogproba = oldlogproba.to(args.device)
        advantages = advantages.to(args.device)
        returns = returns.to(args.device)

        for i_epoch in range(int(args.num_epoch * batch_size / args.minibatch_size)):
            # sample from current batch
            minibatch_ind = np.random.choice(batch_size, args.minibatch_size, replace=False)
            minibatch_observations = observations[minibatch_ind]
            minibatch_actions = actions[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_newlogproba = network.get_logproba(minibatch_observations, minibatch_actions)
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = network._forward_critic(minibatch_observations).flatten()

            assert minibatch_oldlogproba.shape == minibatch_newlogproba.shape
            ratio = torch.exp(minibatch_newlogproba - minibatch_oldlogproba)
            assert ratio.shape == minibatch_advantages.shape
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1 - clip_now, 1 + clip_now) * minibatch_advantages
            loss_surr = - torch.mean(torch.min(surr1, surr2))

            # not sure the value loss should be clipped as well
            # clip example: https://github.com/Jiankai-Sun/Proximal-Policy-Optimization-in-Pytorch/blob/master/ppo.py
            # however, it does not make sense to clip score-like value by a dimensionless clipping parameter
            # moreover, original paper does not mention clipped value
            if args.lossvalue_norm:
                minibatch_return_6std = 6 * minibatch_returns.std()
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2)) / minibatch_return_6std
            else:
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

            loss_entropy = torch.mean(torch.exp(minibatch_newlogproba) * minibatch_newlogproba)

            total_loss = loss_surr + args.loss_coeff_value * loss_value + args.loss_coeff_entropy * loss_entropy
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if args.schedule_clip == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            clip_now = args.clip * ep_ratio

        if args.schedule_adam == 'linear':
            ep_ratio = 1 - (i_episode / args.num_episode)
            lr_now = args.lr * ep_ratio
            # set learning rate
            # ref: https://stackoverflow.com/questions/48324152/
            for g in optimizer.param_groups:
                g['lr'] = lr_now

        if i_episode % args.log_num_episode == 0:
            mean_reward = (torch.sum(rewards) / memory.num_episode).data
            mean_step = len(memory) // memory.num_episode
            print('Finished episode: {} | Reward: {:.4f} | total_loss = {:.4f} = {:.4f} + {} * {:.4f} + {} * {:.4f}' \
                  .format(i_episode, mean_reward, total_loss.cpu().data, loss_surr.cpu().data,
                          args.loss_coeff_value, loss_value.cpu().data, args.loss_coeff_entropy, loss_entropy.cpu().data), end=' | ')
            print('Step: {:d}'.format(mean_step))
            writer.add_scalar('reward', mean_reward, i_episode)
            writer.add_scalar('total_loss', total_loss.cpu().data, i_episode)
            torch.save(network.state_dict(), model_dir + 'network_{}.pth'.format(i_episode))

    if asyn:
        sampler.close()
    else:
        sampler.envs.close()

if __name__ == '__main__':
    main(args)
