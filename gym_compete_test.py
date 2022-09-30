import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gym
import gym_compete
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    PPOPolicy,
    DQNPolicy,
    MultiAgentPolicyManager
)
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.distributions import Independent, Normal

from typing import Any, Dict, List, Optional, Tuple, Union
from gym_compete_wrapper import gym_compete_wrapper
from raw_wrapper import raw_env
from random_policy import RandomPolicy
from tianshou.policy import MultiAgentPolicyManager
from tianshou.trainer import onpolicy_trainer

env_id = 'you-shall-not-pass-humans-v0'
def get_env():
    env = gym.make(env_id)
    env = raw_env(env)
    env = gym_compete_wrapper(env)
    # env = PettingZooEnv(env)
    return env


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )
    parser.add_argument(
        '--n-pistons',
        type=int,
        default=3,
        help='Number of pistons(agents) in the env'
    )
    parser.add_argument('--n-step', type=int, default=100)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--episode-per-collect', type=int, default=16)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--agent-num', type=int, default=2)


    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    parser.add_argument('--render', type=float, default=0.001)


    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    optims: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    # observation_space = [agent1's obs, agent2's obs]
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space

    # agent1's obs_space = agent2's obs_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    if agents is None:
        agents = []
        optims = []
        for _ in range(args.agent_num):
            net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
            actor = ActorProb(
                net, args.action_shape, max_action=args.max_action, device=args.device
            ).to(args.device)
            critic = Critic(
                Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device),
                device=args.device
            ).to(args.device)
            actor_critic = ActorCritic(actor, critic)
            # orthogonal initialization
            for m in actor_critic.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
            optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    
            # replace DiagGuassian with Independent(Normal) which is equivalent
            # pass *logits to be consistent with policy.forward
            def dist(*logits):
                return Independent(Normal(*logits), 1)
    
            agent = PPOPolicy(
                actor,
                critic,
                optim,
                dist,
                discount_factor=args.gamma,
                max_grad_norm=args.max_grad_norm,
                eps_clip=args.eps_clip,
                vf_coef=args.vf_coef,
                ent_coef=args.ent_coef,
                reward_normalization=args.rew_norm,
                advantage_normalization=args.norm_adv,
                recompute_advantage=args.recompute_adv,
                dual_clip=args.dual_clip,
                value_clip=args.value_clip,
                gae_lambda=args.gae_lambda,
                action_space=env.action_space,
            )
            agents.append(agent)
            optims.append(optim)

    policy = MultiAgentPolicyManager(agents, env)
    return policy, optims, env.agents


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    optims: Optional[List[torch.optim.Optimizer]] = None,
) -> Tuple[dict, BasePolicy]:

    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy, optim, agents = get_agents(
        args, agents=agents, optims=optims
    )

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, 'gym_compete', 'dqn')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        # if hasattr(args, 'model_save_path'):
        #     model_save_path = args.model_save_path
        # else:
        #     model_save_path = os.path.join(
        #         args.logdir, 'gym_compete', 'dqn', 'policy.pth'
        #     )
        # torch.save(
        #     policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
        # )
        return 

    def stop_fn(mean_rewards):
        # return mean_rewards >= args.win_rate
        return False 
        
    def train_fn(epoch, env_step):
        # [agent.set_eps(args.eps_train) for agent in policy.policies.values()]
        # policy.policies[agents[args.agent_id - 1]].set_eps(args.ep_train)
        # policy.policies.values()[args.agent_id-1].set_eps(args.ep_train)
        return

    def test_fn(epoch, env_step):
        # [agent.set_eps(args.eps_test) for agent in policy.policies.values()]
        # policy.policies[agents[args.agent_id - 1]].set_eps(args.ep_test)
        # policy.policies.values()[args.agent_id-1].set_eps(args.ep_test)
        return 


    def reward_metric(rews):
        return rews[:, args.agent_id - 1]

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        resume_from_log=args.resume

    )

    return result, policy


def watch(
    args: argparse.Namespace = get_args(),
    policy: Optional[BasePolicy] = None
) -> None:
    env = DummyVectorEnv([get_env])
    # policy, optim, agents = get_agents(
    #     args, agents
    # )
    policy.eval()
    # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")
