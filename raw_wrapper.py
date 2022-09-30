import numpy as np
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

class raw_env(AECEnv):
    metadata = {
        "render_modes": ["rgb_array"],
        "name": "kick_and_defend_v0",
        "is_parallelizable": False,
        "render_fps": 30,
    }
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.agents = [str(i.id) for _, i in env.agents.items()]
        self.possible_agents = self.agents[:]
        self.n_agents = env.n_agents

        self.action_spaces =  {str(i.id): i.action_space for _,i in env.agents.items()}
        self.observation_spaces =  {str(i.id): i.observation_space for _,i in env.agents.items()}
        # self.observation_spaces = {
        #     str(i.id): spaces.Dict(
        #         {
        #             "observation": i.observation_space,
        #             "action_mask": spaces.Box(low=0, high=1, shape=(17,), dtype=np.int8),
        #         }
        #     )
        #     for _,i in env.agents.items()
        # }
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self.rewards = {i: 0 for i in self.agents}

    def observe(self, agent):
        return {"observation": self.env.agents[int(agent)]._get_obs()}
        # return {"observation": self.env.agents[int(agent)]._get_obs(), "action_mask": np.zeros(17, "int8")}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

# obses, rews, done, infos
    def step(self, action):
        ret = self.env.step(action)
        next_agent = self._agent_selector.next()
        self.agent_selection = next_agent
        rewards = ret[1]
        dones = ret[2]
        self.rewards[self.agents[0]] = rewards[0]
        self.rewards[self.agents[1]] = rewards[1]
        self.dones[self.agents[0]] = dones[0]
        self.dones[self.agents[1]] = dones[1]
        return ret
    
    def reset(self, seed=None, return_info=False, options=None):
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.dones = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()
        return self.env.reset()

    def render(self, mode):
        return self.env.render(mode)

    def close(self):
        pass
