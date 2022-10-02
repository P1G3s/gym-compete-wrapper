from abc import ABC
from typing import Any, Dict, List, Tuple, Union

import gym.spaces
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper


class gym_compete_wrapper(AECEnv, ABC):
    def __init__(self, env: BaseWrapper):
        super().__init__()
        self.env = env
        # do not step until all the actions are made
        self.action_available = 0
        self.action_made = []
        # agent idx list
        self.agents = self.env.possible_agents
        self.agent_idx = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_idx[agent_id] = i

        self.rewards = [0] * len(self.agents)

        # Get first observation space, assuming all agents have equal space
        self.observation_space: Any = self.env.observation_space(self.agents[0])

        # Get first action space, assuming all agents have equal space
        self.action_space: Any = self.env.action_space(self.agents[0])

        assert all(self.env.observation_space(agent) == self.observation_space
                   for agent in self.agents), \
            "Observation spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_observations wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_observations(env)`"

        assert all(self.env.action_space(agent) == self.action_space
                   for agent in self.agents), \
            "Action spaces for all agents must be identical. Perhaps " \
            "SuperSuit's pad_action_space wrapper can help (useage: " \
            "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.reset()

    def reset(self, *args: Any, **kwargs: Any) -> Union[dict, Tuple[dict, dict]]:
        self.env.reset(*args, **kwargs)
        observation, _, _, info = self.env.last(self)
        observation_dict = {
            'agent_id': self.env.agent_selection,
            'obs': observation['observation'],
        }

        if "return_info" in kwargs and kwargs["return_info"]:
            return observation_dict, info
        else:
            return observation_dict

    def step(self, action: Any) -> Tuple[Dict, List[int], bool, Dict]:
        self.env.agent_selection = self.env._agent_selector.next()
        self.action_available += 1
        self.action_made.append(action)
        if (self.action_available < len(self.agents)):
            observation, rew, done, info = self.env.last()
            obs = {'agent_id': self.env.agent_selection, 'obs': observation['observation']}
            for agent_id, reward in self.env.rewards.items():
                self.rewards[self.agent_idx[agent_id]] = reward
            return obs, self.rewards, done, info

        observation, rew, done, info = self.env.last()
        obs = {'agent_id': self.env.agent_selection, 'obs': observation['observation']}
        for agent_id, reward in self.env.rewards.items():
            self.rewards[self.agent_idx[agent_id]] = reward
        self.env.step(self.action_made)

        # reset actions that have been made
        self.action_available = 0
        self.action_made = []
        return obs, self.rewards, done, info

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        try:
            self.env.seed(seed)
        except NotImplementedError:
            self.env.reset(seed=seed)

    def render(self, mode: str = "human") -> Any:
        return self.env.render(mode)
