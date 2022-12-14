import argparse
import os

import gym
import gym_compete
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy, PPOPolicy, MultiAgentPolicyManager
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.common import ActorCritic, Net

from tianshou.utils import TensorboardLogger
from torch.distributions import Independent, Normal

from typing import List, Optional, Tuple, Any, Dict
import types
from tianshou.data import Batch


# from tianshou.data import Collector
from collector import Collector
from expert_policy import ExpertPolicy
from gym_compete_wrapper import gym_compete_wrapper
from raw_wrapper import raw_env
from tianshou.trainer import onpolicy_trainer

# Agent 0 (red)   --  blocker
# Agent 1 (green) --  walker
env_id = "you-shall-not-pass-humans-v0"
expert_id = 1
expert_log_path = "./expert_model/you-shall-not-pass/agent{}.pkl".format(expert_id)
# env_id = "kick-and-defend-v0"

# Solves the conflit of both pytorch and tensorflow trying to use the same gpu
# by force pytorch to use another gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
torch.cuda.device_count()

def get_env():
    env = gym.make(env_id)
    env = raw_env(env)
    env = gym_compete_wrapper(env)
    return env


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--step-per-epoch", type=int, default=4096)
    # how many epochs until policy1 updates(load policy0)
    parser.add_argument("--epoch-per-update", type=int, default=5)
    parser.add_argument("--step-per-collect", type=int, default=4096)
    parser.add_argument("--repeat-per-collect", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--training-num", type=int, default=2)
    parser.add_argument("--test-num", type=int, default=2)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--agent-num", type=int, default=2)
    parser.add_argument("--lr-decay", type=int, default=1)
    # temparary assignment
    parser.add_argument("--next-update-epoch", type=int, default=0)

    parser.add_argument(
        "--gamma", type=float, default=0.995, help="a smaller gamma favors earlier win"
    )
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, " "watch the play of pre-trained models",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # ppo special
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--rew-norm", type=int, default=0)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--render", type=float, default=0.001)
    parser.add_argument("--bound-action-method", type=str, default="clip")
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
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )

    # agent1's obs_space = agent2's obs_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]

    if agents is None:
        agents = []
        optims = []
        for i in range(args.agent_num):
            # Override blocker with expert policy
            if i == expert_id:
                agent = ExpertPolicy(env_id, env, expert_log_path, args.device)
                optim = None
            # Normal agent to be trained
            else:
                net = Net(
                    args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device
                )
                actor = ActorProb(
                    net, args.action_shape, max_action=args.max_action, device=args.device
                ).to(args.device)
                critic = Critic(
                    Net(
                        args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device
                    ),
                    device=args.device,
                ).to(args.device)

                for m in list(actor.modules()) + list(critic.modules()):
                    if isinstance(m, torch.nn.Linear):
                        # orthogonal initialization
                        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        torch.nn.init.zeros_(m.bias)
                # do last policy layer scaling, this will make initial actions have (close to)
                # 0 mean and std, and will help boost performances,
                # see https://arxiv.org/abs/2006.05990, Fig.24 for details
                for m in actor.mu.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.zeros_(m.bias)
                        m.weight.data.copy_(0.01 * m.weight.data)

                optim = torch.optim.Adam(
                    list(actor.parameters()) + list(critic.parameters()), lr=args.lr
                )

                lr_scheduler = None
                if args.lr_decay:
                    # decay learning rate to 0 linearly
                    max_update_num = (
                        np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
                    )

                    lr_scheduler = LambdaLR(
                        optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
                    )

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
                    lr_scheduler=lr_scheduler
                    # action_bound_method=args.bound_action_method,
                )

            # Override agent1's learn function for self-play
            # if i == 1:
            #     def new_learn(
            #         self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
            #     ) -> Dict[str, List[float]]:
            #         return {"loss": 0, "loss/clip": 0, "loss/vf": 0, "loss/ent": 0}

            #     agent.learn = types.MethodType(new_learn, agent)

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

    policy, optim, agents = get_agents(args, agents=agents, optims=optims)

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, "gym_compete", "ppo")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, save_interval=args.save_interval)
    args.next_update_epoch = args.epoch_per_update

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        for i in policy.policies:
            ckpt_path = os.path.join(
                    log_path,
                    "checkpoint{}.pth".format(i))
            if os.path.exists(ckpt_path):
                print("Successfully loaded checkpoint")
                checkpoint = torch.load(
                        ckpt_path,
                        map_location=args.device)
                policy.policies[i].load_state_dict(checkpoint["model"])
                policy.policies[i].optim.load_state_dict(
                        checkpoint["optim"])
            else:
                print("Failed to load {}".format(ckpt_path))

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        print("-----saving checkpoint-----")
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        for i in policy.policies:
            if isinstance(policy.policies[i], ExpertPolicy):
                print("no checkpoint for expert policy{}".format(i))
                continue
            ckpt_path = os.path.join(
                args.logdir, "gym_compete", "ppo", "checkpoint{}.pth".format(i)
            )
            torch.save(
                {
                    "model": policy.policies[i].state_dict(),
                    "optim": optim[int(i)].state_dict(),
                },
                ckpt_path,
            )
            model_save_path = os.path.join(
                args.logdir, "gym_compete", "ppo", "policy{}.pth".format(i)
            )
            torch.save(policy.policies[i].state_dict(), model_save_path)
        return ckpt_path

    def stop_fn(mean_rewards):
        # return mean_rewards >= args.win_rate
        return False

    def train_fn(epoch, env_step):
        if epoch > args.next_update_epoch:
            args.next_update_epoch += args.epoch_per_update
            print("###############")
            print("update policy 1")
            print("###############")
            model_load_path = os.path.join(
                args.logdir, "gym_compete", "ppo", "policy0.pth"
            )
            policy.policies["1"].load_state_dict(torch.load(model_load_path))
        # [agent.set_eps(args.eps_train) for agent in policy.policies.values()]
        return

    def test_fn(epoch, env_step):
        [agent.set_eps(args.eps_test) for agent in policy.policies.values()]
        return

    def reward_metric(rews):
        return rews[:, 0]

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
        # stop_fn=stop_fn,
        # train_fn=train_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_checkpoint_fn=save_checkpoint_fn,
        step_per_collect=args.step_per_collect,
        # episode_per_collect=args.episode_per_collect,
    )

    return result, policy


def watch(
    args: argparse.Namespace = get_args(), policy: Optional[BasePolicy] = None
) -> None:
    env = DummyVectorEnv([get_env])
    if not policy:
        # init policy
        policy, _, _ = get_agents(args)
        # load policy
        for i in policy.policies:
            model_load_path = os.path.join(
                args.logdir, "gym_compete", "ppo", "policy{}.pth".format(i)
            )
            policy.policies[i].load_state_dict(torch.load(model_load_path))
    policy.eval()
    collector = Collector(policy, env, exploration_noise=True)
    result = collector.collect(n_episode=20, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")
