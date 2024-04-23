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

        self.selected_ids = None
        self.selected_poilces = None

    def prepare(self, multi_agent_ids, seed=None):
        if seed is not None:
            random.seed(seed)
        self.selected_ids = [
            random.choice(list(self._policies.keys())) for _ in range(len(multi_agent_ids))
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
        if self.selected_ids is None:
            self.prepare([0])
        # assert len(observations) == len(self.selected_poilces), f"{len(observations)}  != {len(self.selected_poilces)}"
        actions = []
        for pi, obs, prev_r in zip(self.selected_poilces, observations, prev_rewards):
            actions.append(pi.step(obs, prev_r))

        return actions[0]


# def build_focal_population(
#     ckpt_paths, policy_ids, scale
# ) -> Iterator[Mapping[str, policy_lib.Policy]]:
#   """Builds a population from the specified saved models.

#   Args:
#     ckpt_paths: path where agent policies are stored
#     policy ids: policy ids for each agent

#   Yields:
#     A mapping from policy id to policy required to build focal population for evaluation.
#   """
#   with contextlib.ExitStack() as stack:
#     yield {
#         p_id: stack.enter_context(DownsamplingPolicyWraper(EvalPolicy(ckpt_paths, p_id), scale))
#         for p_id in policy_ids
#     }

mypop = None

def init():
    # ray.init()
    my_path = os.path.dirname(os.path.abspath(__file__))
    my_path = os.path.join(my_path, "pd_policy")

    config_file = f'{my_path}/params.json'
    f = open(config_file)
    configs = json.load(f)
    # if args.eval_on_scenario:
    #     scenario = args.scenario
    # else:
        # scenario = configs['env_config']['substrate']
    scaled = configs['env_config']['scaled']

    # if args.create_videos:
    #     video_dir = args.video_dir
    # else:
    #     video_dir = None
        
    # policies_path = args.policies_dir
    policies_path = os.path.join(my_path, "checkpoint_001600", "policies")
    roles = configs['env_config']['roles']
    policy_ids = [f"agent_{i}" for i in range(len(roles))]
    names_by_role = defaultdict(list)
    for i in range(len(policy_ids)):
        names_by_role[roles[i]].append(policy_ids[i])

    # Build population and evaluate
    # with build_focal_population(policies_path, policy_ids, scaled) as population:
    #     results = evaluation.evaluate_population(
    #         population=population,
    #         names_by_role=names_by_role,
    #         scenario=scenario,
    #         num_episodes=args.num_episodes,
    #         video_root=video_dir)  
    # return results, scenario
    global mypop
    print("init mypop, before assign", mypop)
    # mypop = 1
    mypop = Population(policies_path, policy_ids, scaled)
    

init()

def my_controller(observation, action_space, is_act_continuous=False):
    global mypop
    # print("obs", observation)
    act = mypop.step([observation], [observation["REWARD"]])
    return [act]
