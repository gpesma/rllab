from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from .mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from math import hypot
from math import log10, floor
from time import sleep

class ThrowerEnv(MujocoEnv, Serializable):
    

    FILE = 'thrower.xml'
    ORI_IND = 2

    
    def __init__(
            self,
            #ctrl_cost_coeff=1e-2,
            *args, **kwargs):
        #self.ctrl_cost_coeff = ctrl_cost_coeff
        super(ThrowerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self._ball_hit_ground = False
        self._ball_hit_location = None

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            #self.get_body_com("r_wrist_roll_link").flat,
            #self.get_body_com("goal"),
            self.get_body_com("ball").flat
        ]).reshape(-1)


    def step(self, action):
        #self.render()
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        ball_xy = self.get_body_com("ball")[:2]
        goal_xy = self.get_body_com("goal")[:2]

        self._ball_hit_ground = False
        reward = 0
        if not self._ball_hit_ground and self.get_body_com("ball")[2] < -0.25:
            self._ball_hit_ground = True
            self._ball_hit_location = self.get_body_com("ball")

        if self._ball_hit_ground:
            ball_hit_xy = self._ball_hit_location[:2]
            reward = (1 -  hypot(ball_hit_xy[0]-goal_xy[0], ball_hit_xy[1]-goal_xy[1])) * 100
        else:
            reward = 0

        done = False

        if reward > 0:
            done = True
            
        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths):
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)

