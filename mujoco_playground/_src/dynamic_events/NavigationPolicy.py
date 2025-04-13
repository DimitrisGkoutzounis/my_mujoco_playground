from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco.mjx._src import math

import mujoco
import numpy as np

from mujoco_playground._src.dynamic_events.arm_mujoco.src.Robot  import RobotGo2
from mujoco_playground._src.dynamic_events.arm_mujoco.src.Arm  import Arm
from mujoco_playground._src.dynamic_events.arm_mujoco.src.Perception import Perception
from mujoco_playground._src.dynamic_events.Locomotion_Controller import Locomotion_Controller
from mujoco_playground._src.dynamic_events.Navigator import Navigator

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
import mujoco_playground._src.dynamic_events.base as go2_base
from mujoco_playground._src.dynamic_events import new_go2_constants

from ml_collections import config_dict
from etils import epath

# NavigationPolicy Class: Defines the navigation policy, TBD
#  Until now: it has Navigator() which decides randomly the vel cmds


# TBD: fix default_config
def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.002,
      episode_length=1000,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
    #   soft_joint_pos_limit_factor=0.95,
    #   noise_config=config_dict.create(
    #       level=1.0,  # Set to 0.0 to disable noise.
    #       scales=config_dict.create(
    #           joint_pos=0.03,
    #           joint_vel=1.5,
    #           gyro=0.2,
    #           gravity=0.05,
    #           linvel=0.1,
    #       ),
    #   ),
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
            #   # Feet.
            #   feet_clearance=-2.0,
            #   feet_height=-0.2,
            #   feet_slip=-0.1,
            #   feet_air_time=0.1,
          ),
          tracking_sigma=0.25,
        #   max_foot_height=0.1,
      ),
    #   pert_config=config_dict.create(
    #       enable=False,
    #       velocity_kick=[0.0, 3.0],
    #       kick_durations=[0.05, 0.2],
    #       kick_wait_times=[1.0, 3.0],
    #   ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          a=[1.5, 0.8, 1.2],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
  )



class NavigationPolicy(go2_base.Go2NavEnv):
    def __init__(
        self,
        # default_angles: np.ndarray = np.zeros(3),
        # action_scale: float = 0.5,
        # vel_scale_x: float = 1.5,
        # vel_scale_y: float = 0.8,
        # vel_scale_rot: float = 2 * np.pi, 
        t_last_cmd: float = 0.0,
        ctrl_dt: float = 0.01, # This affect the Locomotion COntroller
        sim_dt : float = 0.002, # model.opt.timestep
        n_substeps : int = 5, 
        _ONNX_DIR : epath.Path = epath.Path(__file__).parent.parent.parent / "experimental/sim2sim"/ "onnx", # Modified this (_HERE.parent.parent), two folders back before access .onnx
        config: config_dict.ConfigDict = default_config(),
        config_overrides = None,
        xml_path = new_go2_constants.UR5E_GO2_SCENE
       ):
        super().__init__(xml_path=xml_path, config=config, config_overrides=config_overrides )

        self._config = config
        self._n_substeps = n_substeps #int(round(ctrl_dt / sim_dt))
        # self._default_angles = default_angles
        # self._action_scale = action_scale
        self._counter = 0
        self.t_last_cmd = 0.0
        self.dt_new_cmd = 0.34 # HERE TBD ~1/30(vision fps)

        # MuJoCo model-data
        self.model = self._mj_model #TODO # Access mujoco model
        self.mujoco_data = mujoco.MjData(self.model)

        self._ONNX_DIR = _ONNX_DIR
        
        self._post_init()

    def _post_init(self) -> None:

        self.robot_go2 = RobotGo2()

        # Constructor Locomotion_Controller
        self.Loc_Ctrl = Locomotion_Controller(
        locomotion_policy_path=(self._ONNX_DIR / "go2_policy_galloping.onnx").as_posix(),
        # Pass only qpos regarding Go2's joints, NOT base's x,y,z,quat 
        default_angles=np.array(self.model.keyframe("home").qpos[self.robot_go2.i_start_qpos:self.robot_go2.i_end_qpos]),
        # Pass different n_substeps if need. Works: int(round(0.01 / 0.002))
        n_substeps= 5, # int(round(ctrl_dt / sim_dt)) 0.01/0.002 = 5
        action_scale=0.5,
        vel_scale_x=1.5,
        vel_scale_y=0.8,
        vel_scale_rot=2 * np.pi,
        locomotion_cmd=np.zeros(3)) # Define which Navigator-Boss the locomotion cotroller will have.

        # Policy instances
        data = mujoco.MjData(self._mj_model)
        # Initialize Go2 pos,quat at init_*
        self.robot_go2.get_CoM_pos(data=data)
        self.robot_go2.init_pc =  self.robot_go2.pc
        self.robot_go2.init_xquat =  self.robot_go2.xquat
        print("From post_init_: robots pos = \n", self.robot_go2.init_pc) 

        # Arm
        self.arm = Arm()
        # Perception #TODO
        # self.perception = Perception()  #, perception_cont
        # Added this to decide the vel Cmds: in future, it will come from Navigation Policy
        self.Navigator_ = Navigator()
        
        print("NavigationPolicy: Initialized")

    def get_action(self, model, data) -> np.ndarray:
        """Returns zero action (does nothing)."""
        return np.zeros_like(self._default_angles, dtype=np.float32)
        
    # def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    #     self._counter += 1
    #     if self._counter % self._n_substeps == 0:
    #         data.ctrl[:] = self.get_action(model, data)
    #         print(f"Simulation time: {data.time:.4f}")
    #         print(f"Action: {data.ctrl}")
    #         print(f"Sensor data: {data.sensor('local_linvel').data}")
    #         print(f"Sensor data: {data.sensor('gyro').data}")

    def set_locomotion_ctrl(self, LocCtrl__):
        self.Loc_Ctrl = LocCtrl__


    def step(self, model, data):
        # print("In step")
        # print("Data time", data.time)
        # print("t_last_cmd", self.t_last_cmd)
        # print("dt_new_cmd", self.dt_new_cmd)

        # Set new vel cmd per dt_new_cmd
        if (data.time - self.t_last_cmd) >= self.dt_new_cmd:
            print("New cmd at:",data.time)

    #################################################################
            # Update locomotion cmd, from whatever(in random) the Navigator decides
            # TBD: replace navigator from you policy decision
            self.Loc_Ctrl.update_locomotion_cmd(self.Navigator_.generate_command_norot()) 
    #################################################################

            # Reset t_last_cmd 
            self.t_last_cmd = data.time

        # Move arm
        self.arm.control_Cb(model=model, data=data)
        
        # Perceive
        # self.perception.get_rgbd_auto_AOI(model=model, data=data, perception_context=self.context)
        
        # Execute the locomotion cmd
        self.Loc_Ctrl.exec_locomotion_control(model=model, data=data, robot=self.robot_go2)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        xpos = self.robot_go2.init_pc # or set to []
        xquat = self.robot_go2.init_xquat # or set to init [] manually
        
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        # xpos = xpos.at[0:2].set(xpos[0:2] + dxy)
        xpos = jp.array(self.robot_go2.init_pc)  # Convert to a JAX array
        xpos = xpos.at[0:2].set(xpos[0:2] + dxy)   # Use jp’s update (no np.array conversion)



        # #TODO check if yaw needs also a key
        # ???
        # rng, key = jax.random.split(rng)
        # yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        data = mjx_env.init(self.mjx_model, xpos=xpos, xquat=xquat) #, ctrl=qpos[7:]

        rng, key1, key2 = jax.random.split(rng, 3)
        time_until_next_cmd = jax.random.exponential(key1) * 5.0
        steps_until_next_cmd = jp.round(time_until_next_cmd / self.dt).astype(
            jp.int32
        )
        cmd = jax.random.uniform(
            key2, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a
        )

        info = {
            "rng": rng,
            "command": cmd,
            "steps_until_next_cmd": steps_until_next_cmd,
            "last_act": jp.zeros(2), # vel x,y
            "last_last_act": jp.zeros(2),  # vel x,y
            "last_perceive": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
        }        
        # return super().reset(rng)
        metrics = {}
        metrics["a_metric"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)


    def _get_reward(self):
        pass
    def _get_termination(self):
        pass
        # Here petributions
        # Placeholder petrubations

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_upvector(data)[-1] < 0.0
        return fall_termination

    def _reset_if_outside_bounds(self, state: mjx_env.State) -> mjx_env.State:
        print("Define me : _reset_if_outside_bounds")
        pass

    # Without noise 
    # #TODO add noise
    # #TODO dict info?
    def _get_obs(
             self, data: mjx.Data, info: dict[str, Any]
    ) -> Dict[str, jax.Array]:
        # Update state from mujoco data
        self.robot_go2.get_CoM_pos(data)
        # Obs stored at self.robot_go2.pc, self.robot_go2.xquat
        robot_pos = self.robot_go2.pc
        robot_quat = self.robot_go2.xquat

        state = jp.hstack([
            robot_pos,  # 3
            robot_quat,  # 4
            info["last_act"],  # 2 or 3 if yaw
            info["command"],  # 2 or 3 if yaw
        ])
