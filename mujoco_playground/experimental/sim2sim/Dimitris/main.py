# ==============================================================================
# ==============================================================================

# Hey, I modified Copyright 2025 DeepMind Technologies Limited
# for an upcoming project regarding navigation policies for Go2 Unitree's quadrupedal robot.

# ==============================================================================
# ==============================================================================
# Copyright 2025 DeepMind Technologies Limited
#
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

"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
import pygame

pygame.init()
pygame.display.set_mode((1, 1))  # Ensures the video system is initialized

from mujoco_playground._src.locomotion.go2 import go2_constants
from mujoco_playground._src.locomotion.go2.base import get_assets
from mujoco_playground.experimental.sim2sim.Dimitris.LocomotionPolicy import LocomotionPolicy
from mujoco_playground.experimental.sim2sim.Dimitris.TrajectoryGenerator import TrajectoryGenerator
from mujoco_playground.experimental.sim2sim.Dimitris.TrajectoryPolicy import TrajectoryPolicy 


_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE.parent / "onnx"

def main(model=None, data=None):
  
  #set mujoco control to None
  mujoco.set_mjcb_control(None)

  # Load xml from the path
  model = mujoco.MjModel.from_xml_path(
      go2_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )
  
  #load the data, from now, always use the data object.
  data = mujoco.MjData(model)

  #set the default keyframe
  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.02 #50Hz
  sim_dt = 0.001 #250Hz
  n_substeps = int(round(ctrl_dt / sim_dt)) #for each control step, how many simulation steps are taken(5 in this case)
  model.opt.timestep = sim_dt # set the simulation timestep


  locomotion_policy = LocomotionPolicy(
      policy_path=(_ONNX_DIR / "go2_policy_galloping.onnx").as_posix(), #loads the policy
      default_angles=np.array(model.keyframe("home").qpos[7:]), #home position
      n_substeps=n_substeps,  #number of substeps
      #--- action scales ----#
      action_scale=0.5, 
      vel_scale_x=1.5, 
      vel_scale_y=0.8,
      vel_scale_rot=2 * np.pi,
  )
  
  generator = TrajectoryGenerator()
  
  
    
  
  trajectory_policy = TrajectoryPolicy(
    locomotion_policy=locomotion_policy,
    trajectory_generator=generator,
    n_substeps=n_substeps,
  )
  
  #set the high-level controller
  mujoco.set_mjcb_control(trajectory_policy.trajectory_controller)   
  
  
  
  return model, data


if __name__ == "__main__":
  print("Running with random velocity controller...")

  pygame.event.pump()
  viewer.launch(loader=main)
