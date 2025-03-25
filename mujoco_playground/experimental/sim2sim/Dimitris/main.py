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
from mujoco_playground.experimental.sim2sim.Dimitris.TrajectoryPolicy import TrajectoryPolicy
from mujoco_playground.experimental.sim2sim.Dimitris.generator import ShapeGenerator


_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE.parent / "onnx"

def main(model=None, data=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      go2_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 0)

  ctrl_dt = 0.02
  sim_dt = 0.004
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt


  locomotion_policy = LocomotionPolicy(
      policy_path=(_ONNX_DIR / "go2_policy_20250324_230734.onnx").as_posix(), #loads the policy
      default_angles=np.array(model.keyframe("home").qpos[7:]), #home position
      n_substeps=n_substeps,  #number of substeps
      action_scale=0.5, #action scale
      
      #--- action scales ----#
      
      vel_scale_x=1.5, 
      vel_scale_y=0.8,
      vel_scale_rot=2 * np.pi,
  )
  
  
  trajectory_policy = TrajectoryPolicy(
    locomotion_policy=locomotion_policy,
    n_substeps=n_substeps,
    max_vel=5,  # Adjust as needed
)
  # set high-level controller
  
  mujoco.set_mjcb_control(trajectory_policy.controller)       # High-level trajectory
  # mujoco.set_mjcb_control(locomotion_policy.get_control)     # Low-level ONNX locomotion
  
  
  
  return model, data


if __name__ == "__main__":
  print("Running with random velocity controller...")

  pygame.event.pump()
  viewer.launch(loader=main)
