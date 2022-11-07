from typing import Any, Dict, Optional, Union

import numpy as np

from tianshou.data import Batch
from tianshou.policy import BasePolicy
from tf_policy import LSTMPolicy, MlpPolicyValue
import pickle
import tensorflow as tf
import torch


def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params


def setFromFlat(var_list, flat_params):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(
            v,
            tf.reshape(theta[start:start + size], shape)
            ))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})


class ExpertPolicy(BasePolicy):
    def __init__(self, env_id, env, expert_log_path, device):
        super().__init__()
        self.device = device
        if env_id == "kick-and-defend":
            policy_type = "lstm"
        elif env_id == "you-shall-not-pass-humans-v0":
            policy_type = "mlp"
        scope = "policy_expert"

        # Init policy
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,)
        sess = tf.Session(config=tf_config)
        sess.__enter__()
        if policy_type == "lstm":
            self.policy = LSTMPolicy(scope=scope, reuse=False,
                                     ob_space=env.observation_space,
                                     ac_space=env.action_space,
                                     hiddens=[128, 128], normalize=True)
        else:
            self.policy = MlpPolicyValue(scope=scope, reuse=False,
                                         ob_space=env.observation_space,
                                         ac_space=env.action_space,
                                         hiddens=[64, 64], normalize=True)
        # Load params
        param = load_from_file(expert_log_path)
        setFromFlat(self.policy.get_variables(), param)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:

        action = self.policy.act(stochastic=True, observation=batch.obs)
        action = torch.tensor(action).to(self.device)
        return Batch(act=action)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        return {}
