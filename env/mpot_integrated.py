# -*- coding:utf-8  -*-

import copy
import random
import time

from env.simulators.game import Game
from env.obs_interfaces.observation import *
from meltingpot import substrate
from meltingpot.utils.policies import policy
from meltingpot import scenario
import immutabledict
import dm_env
# from overcooked_ai_py.mdp.actions import Action
# from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
# from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
# from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from utils.discrete import Discrete
# from gym.spaces.discrete import Discrete

import gym
import numpy as np
import pygame

__all__ = ['MPot_Integrated']

GRID1= """XXXPX
O X P
O X X
D X X
XXXSX"""


COLORS = {
    'red': [255,0,0],
    'light red': [255, 127, 127],
    'green': [0, 255, 0],
    'blue': [0, 0, 255],
    'orange': [255, 127, 0],
    'grey':  [176,196,222],
    'purple': [160, 32, 240],
    'black': [0, 0, 0],
    'white': [255, 255, 255],
    'light green': [204, 255, 229],
    'sky blue': [0,191,255],
    # 'red-2': [215,80,83],
    # 'blue-2': [73,141,247]
}

pygame.init()
font = pygame.font.Font(None, 18)
def display_text(info, y = 10, x=10, c='black'):
    # display_surf = pygame.display.get_surface()
    debug_surf = font.render(str(info), True, COLORS[c])
    debug_rect = debug_surf.get_rect(topleft = (x,y))
    # display_surf.blit(debug_surf, debug_rect)
    return debug_surf, debug_rect

class MPot_Integrated(Game, DictObservation):
    def __init__(self, conf):
        super(MPot_Integrated, self).__init__(n_player=-1, is_obs_continuous=True, is_act_continuous=False, 
                                              game_name=conf['game_name'], agent_nums=-1, 
                                              obs_type=None)
        self.conf = conf
        self.done = False
        self.step_cnt = 0
        self.max_step = int(conf["max_step"])
        self.num_episode = int(conf["num_episode"])
        self.game_name = self.conf['game_name']
        
        self.debug_mode = self.conf.get("debug_mode", False)

        if conf['scenarios'] is not None:
            self.scenario_pool = conf['scenarios']
        else:
            self.scenario_pool = []
        self.current_scenario = 0
        self.current_episode = 0
        
        # self.game_pool = ['cramped_room_tomato','cramped_room_tomato',
        #                   'forced_coordination_tomato', 'forced_coordination_tomato',
        #                   'soup_coordination', 'soup_coordination']
        # random.shuffle(self.game_pool)
        self.horizon = 400
        self.agent_mapping = [[0,1],[1,0]]*3
        self.player2agent_mapping = None

        self.player_state = None
        self.player_reward = None

        self.reset_map(init=True)
        # self.env.display_states(self.env.state)

        # TODO: What is Action.ALL_ACTIONS? rewrite action_space & obs_space
        # self.action_space = Discrete(len(Action.ALL_ACTIONS))

        self.joint_action_space = self.set_action_space()

        self.init_info = {}
        self.won = {}
        self.n_return = [0] * self.n_player
        self.action_dim = self.get_action_dim()
        self.observation_space = self._setup_observation_space()
        self.input_dimension = None
        self.human_play = False

        # self.render_mode = 'window'     #'console', 'window' or None
        self.render_mode = None

        # if self.render_mode is not None:
        #     self.vis = StateVisualizer()
        #     self.window_size = (500, 500)
        #     self.background = pygame.display.set_mode(self.window_size)

    def _setup_observation_space(self):
        observation_space = self.env.observation_spec()
        obs_space = []
        for s in observation_space:
            if 'WORLD.RGB' in s:
                ss = dict(s)
                del ss['WORLD.RGB']
            else:
                ss = s
            obs_space.append(ss)
            
        return obs_space

    def _state_filter(self, ts):
        """
            Remove global state from observation.
            Only keep it in debug mode.
        """
        if self.debug_mode:
            return ts.observation, ts.reward
        # delete specific keys
        new_state = []

        for s in ts.observation:
            if isinstance(s, immutabledict.immutabledict):
                ss = dict(s)
                if 'WORLD.RGB' in ss:
                    del ss['WORLD.RGB']
            else:
                ss = s
            
            # add step_type here
            ss["STEP_TYPE"] = ts.step_type
            new_state.append(ss)
        return new_state, ts.reward
    
    def _create_scenario_game(self, scenario_id):
        name = f"{self.game_name}_{scenario_id}"
        factory = scenario.get_factory(name)
        num_players = factory.num_focal_players()
        # action_spec = [factory.action_spec()] * num_players
        # reward_spec = [factory.timestep_spec().reward] * num_players
        # discount_spec = factory.timestep_spec().discount
        # observation_spec = dict(factory.timestep_spec().observation)
        # observation_spec['COLLECTIVE_REWARD'] = dm_env.specs.Array(
        #     shape=(), dtype=np.float64, name='COLLECTIVE_REWARD')
        # observation_spec = [observation_spec] * num_players
        env = factory.build()
        return env, num_players

    def _create_substrate_game(self,):
        factory = substrate.get_factory(self.game_name)
        roles = factory.default_player_roles()
        # action_spec = [factory.action_spec()] * len(roles)
        # reward_spec = [factory.timestep_spec().reward] * len(roles)
        # discount_spec = factory.timestep_spec().discount
        # observation_spec = dict(factory.timestep_spec().observation)
        # observation_spec['COLLECTIVE_REWARD'] = dm_env.specs.Array(
        #     shape=(), dtype=np.float64, name='COLLECTIVE_REWARD')
        # observation_spec = [observation_spec] * len(roles)
        env = factory.build(roles)
        return env, len(roles)

    def reset_get_scenario(self, init):
        # if reset_map in __init__, do not count for episode.
        if len(self.scenario_pool) == 0:
            if not init:
                self.current_episode += 1
            return -1
        elif init:
            return self.scenario_pool[0]
        
        self.current_episode += 1
        if self.current_episode > self.num_episode:
            self.current_scenario += 1
            self.current_episode = 1

        print("reset scenario", self.current_scenario)
        scenario = self.scenario_pool[self.current_scenario]
        return scenario

    def reset_map(self, init=False):
        """
            Create Game Environment 
        """
        # Create Game first
        scenario_id = self.reset_get_scenario(init=init)
        if scenario_id == -1:
            self.env, self.n_player = self._create_substrate_game()
        else:
            self.env, self.n_player = self._create_scenario_game(scenario_id)
        self.current_game = self.game_name
        
        # self.player_state, self.player_reward = self._state_filter(self.env.reset())
        timestep = self.env.reset()
        state, reward = self._state_filter(timestep)
        # global_state = self.env.state.to_dict()
        # self.player_state = [global_state for _ in range(self.n_player)]            #share across agents

        # Set all vars in Game
        self.agent_nums = [1] * self.n_player
        self.obs_type = ["dict"] * self.n_player

        print(f'Game: {self.game_name}, Senario:{scenario_id}, Player_num{self.n_player}')
        print(f'Obs Space: {self.env.observation_spec()[0]}, Act Space:{self.env.action_spec()[0]}')
        
        return state, reward, timestep
        # self.all_observes = self.get_all_observes(new_map=True)

    @property
    def last_game(self):
        if len(self.scenario_pool) == 0:
            return self.current_episode == self.num_episode
        return self.current_episode == self.num_episode and self.current_scenario == len(self.scenario_pool)-1

    def reset(self):
        self.step_cnt = 0
        self.done = False

        # self.game_pool = ['cramped_room_tomato','cramped_room_tomato',
        #                   'forced_coordination_tomato', 'forced_coordination_tomato',
        #                   'soup_coordination', 'soup_coordination']
        # self.agent_mapping = [[0,1],[1,0]]*3
        # self.player2agent_mapping = None
        state, reward, timestep = self.reset_map()

        # self.init_info = {"game_pool": self.game_pool, "agent_mapping": self.agent_mapping}
        self.won = {}
        self.n_return = [0] * self.n_player
        # return self.player_state
        return state, reward

    def retrieve_human_play(self, joint_action):
        while True:
            action = None
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = [1,0,0,0,0,0]
                    if event.key == pygame.K_DOWN:
                        action = [0,1,0,0,0,0]
                    if event.key == pygame.K_RIGHT:
                        action = [0,0,1,0,0,0]
                    if event.key == pygame.K_LEFT:
                        action = [0,0,0,1,0,0]
                    if event.key == pygame.K_SPACE:
                        action = [0,0,0,0,0,1]
                    if event.key == pygame.K_TAB:
                        action = [0,0,0,0,1,0]
                    joint_action[0][0] = np.array(action)
                    break
                else:
                    pass
                    # action = [0,0,0,0,1,0]
            if action is not None:
                break
        return joint_action

    def step(self, joint_action):

        # joint_action_decode = self.decode(joint_action)
        # info_before = self.step_before_info(joint_action_decode)
        map_done = self.step_cnt >= self.max_step
        if map_done and not self.last_game:
            next_state, reward, timestep = self.reset_map()
            self.step_cnt = 0
        else:
            timestep = self.env.step(joint_action)
            next_state, reward = self._state_filter(timestep)
        info_after = {
            "step_type": timestep.step_type,
            "discount": timestep.discount,
            "step": self.step_cnt,
            "episode": self.current_episode,
            "scenario": self.current_scenario,
            "n_return": self.n_return
        }
        

        #TODO: to include value of ingradient
        if map_done and not self.last_game:
            info_after['new_map'] = True

            # next_state = next_state.to_dict()
        #     self.player_state = [next_state for _ in range(self.n_player)]
        #     # self.current_state = [next_state for _ in range(self.n_player)]
        #     self.all_observes = self.get_all_observes()

        # recipe_config = self.env.mdp.recipe_config        #config for food cooking time and value
        self.set_n_return(reward)
        self.step_cnt += 1
        done = self.is_terminal()
        # info_after = self.parse_info_after(info_after)
        return next_state, reward, done, {}, info_after

    def step_before_info(self, env_action):
        info = {
            "env_actions": env_action,
            "player2agent_mapping": self.player2agent_mapping
        }

        return info

    def parse_info_after(self, info_after):
        if 'episode' in info_after:
            episode = info_after['episode']
            for out_key, out_value in episode.items():
                if isinstance(out_value, dict):
                    for key, value in out_value.items():
                        if isinstance(value, np.ndarray):
                            info_after['episode'][out_key][key] = value.tolist()
                elif isinstance(out_value, np.ndarray):
                    info_after['episode'][out_key] = out_value.tolist()
                elif isinstance(out_value, np.int64):
                    info_after['episode'][out_key] = int(out_value)

        info_after['state'] = self.env.state.to_dict()
        info_after['player2agent_mapping'] = self.player2agent_mapping

        return info_after

    def is_terminal(self):
        if self.step_cnt >= self.max_step and self.last_game:
            self.done = True

        return self.done

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def get_dict_observation(self, current_state, player_id, info_before):
        return current_state

    def set_action_space(self):
        """ Convert dmlab action space to joint action space
        
            env act (immutabledict({'INVENTORY': Array(shape=(2,), dtype=dtype('float64'), name='INVENTORY'), 
                        'RGB': Array(shape=(88, 88, 3), dtype=dtype('uint8'), name='RGB'), 
                        'READY_TO_SHOOT': Array(shape=(), dtype=dtype('float64'), name='READY_TO_SHOOT'), 
                        'COLLECTIVE_REWARD': Array(shape=(), dtype=dtype('float64'), name='COLLECTIVE_REWARD')}),) 
                (DiscreteArray(shape=(), dtype=int64, name=action, minimum=0, maximum=7, num_values=8),)
        """
        origin_action_space = self.env.action_spec()
        print("original act", origin_action_space)
        new_action_spaces = [[Discrete(space.num_values)] for space in origin_action_space]
        
        return new_action_spaces

    def check_win(self):
        return self.won

    def get_action_dim(self):
        action_dim = 1
        if self.is_act_continuous:
            # if isinstance(self.joint_action_space[0][0], gym.spaces.Box):
            assert NotImplementedError('Continuous action space not implemented')
            return self.joint_action_space[0][0]

        for i in range(len(self.joint_action_space)):
            item = self.joint_action_space[i][0]
            if isinstance(item, Discrete):
                action_dim *= item.n

        return action_dim

    def decode(self, actions):
        return actions
        # assert NotImplementedError('Decode not used in meltingpot')
        joint_action_decode = []
        # joint_action_decode_tmp = []
        # for nested_action in joint_action:
        #     if not isinstance(nested_action[0], np.ndarray):
        #         nested_action[0] = np.array(nested_action[0])
        #     joint_action_decode_tmp.append(nested_action[0].tolist().index(1))

        # #swap according to agent index
        # joint_action_decode_tmp2 = [joint_action_decode_tmp[i] for i in self.player2agent_mapping]

        for a in actions:
            joint_action_decode.append(a)
        # for action_id in joint_action_decode_tmp2:
        #     joint_action_decode.append(Action.INDEX_TO_ACTION[action_id])

        return joint_action_decode

    def set_n_return(self, reward):
        for i in range(self.n_player):
            self.n_return[i] += reward[i]

    def set_seed(self, seed=0):
        np.random.seed(seed)

    def get_all_observes(self, new_map=False):
        all_observes = []
        for player_idx in range(self.n_player):
            mapped_agent_idx = self.player2agent_mapping[player_idx]
            each = copy.deepcopy(self.player_state[mapped_agent_idx])
            if each:
                each["controlled_player_index"] = mapped_agent_idx
                each['new_map'] = new_map
            all_observes.append(each)
        return all_observes
