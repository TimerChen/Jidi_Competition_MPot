# -*- coding:utf-8  -*-
# Time  : 2021/5/31 下午4:14
# Author: Yahui Cui

"""
# =================================== Important =========================================
Notes:
1. this agent is random agent , which can fit any env in Jidi platform.
2. if you want to load .pth file, please follow the instruction here:
https://github.com/jidiai/ai_lib/blob/master/examples/demo
"""


def my_controller(observation, action_space, is_act_continuous=False):
    """
    WARNING: KEEP AN EYE ON THIS FUNNY INPUT PARAMS!
        IF YOU NEED TO WRITE RULES PLEASE MATCH THIS INPUT
    observation: an state which in our project is a DICT
    action_space_list_each: only one action space which is wrapped into a list, :(
    """
    agent_action = []
    for i in range(len(action_space)):
        action_ = sample_single_dim(action_space[i], is_act_continuous)
        agent_action.append(action_)
    return agent_action


def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
    return each
