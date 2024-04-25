from typing import Tuple, List, Any
import dm_env
from meltingpot.utils.policies import policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy import sample_batch

import random
import copy
import torch
import ray
import json
import numpy as np

from collections import defaultdict

import argparse
import os
import cv2
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))

# parser = argparse.ArgumentParser()
# parser.add_argument("-obs_space", default=4, type=int)
# parser.add_argument("-action_space", default=2, type=int)
# parser.add_argument("-hidden_size", default=64, type=int)
# parser.add_argument("-algo", default="dqn", type=str)
# parser.add_argument("-network", default="critic", type=str)
# parser.add_argument("-n_player", default=1, type=int)
# args = parser.parse_args()

# agent = SingleRLAgent(args)
my_path = os.path.dirname(os.path.abspath(__file__))
my_path = os.path.join(my_path, "pd_policy")

# agent.load(critic_net)

sys.path.pop(-1)  # just for safety

_IGNORE_KEYS = [
    "WORLD.RGB",
    "INTERACTION_INVENTORIES",
    "NUM_OTHERS_WHO_CLEANED_THIS_STEP",
    "REWARD",
    "STEP_TYPE"
]

def downsample_observation(array: np.ndarray, scaled) -> np.ndarray:
    """Downsample image component of the observation.
    Args:
      array: RGB array of the observation provided by substrate
      scaled: Scale factor by which to downsaple the observation
    
    Returns:
      ndarray: downsampled observation  
    """
    
    frame = cv2.resize(
            array, (array.shape[0]//scaled, array.shape[1]//scaled), interpolation=cv2.INTER_AREA)
    return frame

class EvalPolicy(policy.Policy):
    """Loads the policies from  Policy checkpoints and removes unrequired observations
    that policies cannot expect to have access to during evaluation.
    """

    def __init__(
        self,
        chkpt_dir: str,
        policy_id: str = sample_batch.DEFAULT_POLICY_ID,
        scale: float = 8,
    ) -> None:

        self._policy_id = policy_id
        policy_path = f"{chkpt_dir}/{policy_id}"
        self._policy = Policy.from_checkpoint(policy_path)
        self._prev_action = 0
        self._scale = scale

    def initial_state(self) -> policy.State:
        """See base class."""

        self._prev_action = 0
        state = self._policy.get_initial_state()
        self.prev_state = state
        return state

    def step(self, observations, prev_reward: Any) -> Tuple[int, policy.State]:
        """See base class."""
        new_obs = {}
        for k in observations:
            if k not in _IGNORE_KEYS:
                new_obs[k] = observations[k]
        observations = new_obs
        # for k in _IGNORE_KEYS:
        #     assert k not in observations

        observations = {
            k: downsample_observation(v, self._scale) if k == "RGB" else v
            for k, v in observations.items()
        }

        # We want the logic to be stateless so don't use prev_state from input
        action, state, _ = self._policy.compute_single_action(
            observations,
            self.prev_state,
            prev_action=self._prev_action,
            prev_reward=prev_reward,
        )

        self._prev_action = action
        self.prev_state = state
        return action

    def close(self) -> None:
        """See base class."""


class Population:
    def __init__(
        self,
        ckpt_paths: str,
        policy_ids: List[str],
        scale: float,
    ):
        self._policies = {
            p_id: EvalPolicy(ckpt_paths, p_id, scale) for p_id in policy_ids
        }
        self.ckpt_paths = ckpt_paths
        self._policy_ids = policy_ids
        self.scale = scale

        self.selected_ids = None
        self.selected_poilces = []

    def _load_policy(self, pid):
        return EvalPolicy(self.ckpt_paths, pid, self.scale)

    def prepare(self, multi_agent_ids, seed=None):
        self.finish()
        if seed is not None:
            random.seed(seed)
        self.selected_ids = [
            random.choice(self._policy_ids) for _ in range(len(multi_agent_ids))
        ]
        # logger.debug(f"Population.prepare: Select {self.selected_ids}")
        self.selected_poilces = [
            self._policies[p_id] for p_id in self.selected_ids
        ]
        for p in self.selected_poilces:
            p.initial_state()

    def finish(self):
        for p in self.selected_poilces:
            p.close()
        del self.selected_poilces
        self.selected_ids = None
        self.selected_poilces = None
        # torch.cuda.empty_cache()

    def step(self, observations: List[Any], prev_rewards: List[Any]):
        # if dm_env.StepType.FIRST
        if observations[0]["STEP_TYPE"] == 0:
            print("init policy")
            self.prepare([0])
        # assert len(observations) == len(self.selected_poilces), f"{len(observations)}  != {len(self.selected_poilces)}"
        actions = []
        for pi, obs, prev_r in zip(self.selected_poilces, observations, prev_rewards):
            actions.append(pi.step(obs, prev_r))

        return actions[0]

mypop = None

def init():
    # ray.init()
    my_path = os.path.dirname(os.path.abspath(__file__))
    my_path = os.path.join(my_path, "pd_policy")

    config_file = f'{my_path}/params.json'
    f = open(config_file)
    configs = json.load(f)
    scaled = configs['env_config']['scaled']

    # TODO: agent path at "pd_policy/checkpoint_000001/"
    policies_path = os.path.join(my_path, "checkpoint_000001", "policies")
    roles = configs['env_config']['roles']
    policy_ids = [f"agent_{i}" for i in range(len(roles))]
    names_by_role = defaultdict(list)
    for i in range(len(policy_ids)):
        names_by_role[roles[i]].append(policy_ids[i])

    # Build population and evaluate
    global mypop
    # mypop = 1
    mypop = Population(policies_path, policy_ids, scaled)
    

init()

def my_controller(observation, action_space, is_act_continuous=False):
    global mypop
    act = mypop.step([observation], [observation["REWARD"]])
    return [act]
