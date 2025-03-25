# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Changda Tian, FORTH.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Go2."""







from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go2 import base as go2_base
from mujoco_playground._src.locomotion.go2 import go2_constants as consts


def default_config() -> config_dict.ConfigDict:
    
    pass
    """Returns the default configuration for the Go2 navigation task."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Tracking.
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.5,
                # Base reward.
                lin_vel_z=-0.5,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                # Other.
                dof_pos_limits=-1.0,
                pose=0.5,
                # Other.
                termination=-1.0,
                stand_still=-1.0,
                # Regularization.
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
                # Feet.
                feet_clearance=-2.0,
                feet_height=-0.2,
                feet_slip=-0.1,
                feet_air_time=0.1,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.1,
        ),
        pert_config=config_dict.create(
            enable=False,
            velocity_kick=[0.0, 3.0],
            kick_durations=[0.05, 0.2],
            kick_wait_times=[1.0, 3.0],
        ),
        command_config=config_dict.create(
            # Uniform distribution for command amplitude.
            a=[1.5, 0.8, 1.2],
            # Probability of not zeroing out new command.
            b=[0.9, 0.25, 0.5],
        ),
    )
    
    
class NavigateShape(go2_base.Go2Env):
    
    """ Learn to track a sqaure trajectory with constant velocity per side."""
    
    
