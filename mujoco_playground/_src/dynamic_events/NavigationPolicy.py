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



class NavigationPolicy:#(go2_base.Go2NavEnv):
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
        # super().__init__(xml_path=xml_path, config=config, config_overrides=config_overrides )

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
        
        
        
        #many times. 
        
        
        # low_level_ste()
        
        
        
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

        info = {
            "t_last_cmd": self.t_last_cmd,
            "sim_time": data.time,
            "locomotion_cmd": self.Loc_Ctrl.locomotion_cmd,
        }

        # Generate the observation.
        # The _get_obs function stacks the robot's position, orientation, last action and command.
        obs = self._get_obs(data, info)

        # Placeholder for reward computation. In the future, we compute the reward based on tracking,
        # obstacle avoidance, view quality of the arm etc.
        reward = jp.array(0.0)

        # Check for termination. For instance, if the robot has fallen over (its up vector has a negative z-component).
        done = self._get_termination(data)

        # Collect additional metrics, if desired.
        metrics = {
            "sim_time": data.time,
            "t_last_cmd": self.t_last_cmd,
            "locomotion_cmd": self.Loc_Ctrl.locomotion_cmd,
        }

        # Return the new state with updated data, observation, reward, termination flag, metrics, and info.
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        xpos = self.robot_go2.init_pc # or set to []
        xquat = self.robot_go2.init_xquat # or set to init [] manually
        
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        xpos = xpos.at[0:2].set(xpos[0:2] + dxy)
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


    def _get_reward(self, model: mujoco.MjModel, data: mujoco.MjData) -> jax.Array:
        # --- Tracking Reward ---
        # Get the measured linear and angular velocities from the sensors.
        # 'local_linvel' should be a 3D vector [v_x, v_y, v_z].
        local_linvel = data.sensor("local_linvel").data  # e.g. using Mujoco sensor
        # 'gyro' is assumed to provide angular velocities; we take the yaw (last element).
        local_gyro = data.sensor("gyro").data

        # Desired velocities come from the locomotion controller command.
        desired_cmd = self.Loc_Ctrl.locomotion_cmd
        # Compute the error (Euclidean norm) for the linear velocities (v_x and v_y).
        error_lin = jp.linalg.norm(local_linvel[:2] - desired_cmd[:2])
        # For the angular velocity (yaw rate) measure absolute error.
        error_ang = jp.abs(local_gyro[-1] - desired_cmd[2])

        # Use scales provided in the configuration.
        reward_tracking = (
            self._config.reward_config.scales.tracking_lin_vel * (-error_lin) +
            self._config.reward_config.scales.tracking_ang_vel * (-error_ang)
        )

        # --- Orientation Reward ---
        # Encourage the robot to maintain an upright posture.
        upvec = self.get_upvector(data)  # assumed to return a vector (e.g., [u_x, u_y, u_z])
        reward_orientation = self._config.reward_config.scales.orientation * upvec[-1]

        # --- Perception-Based Reward ---
        # If a perception module is integrated, use it to evaluate the quality of the arm's view
        # and to measure the proximity of obstacles.
        reward_perception = 0.0
        if hasattr(self, "perception"):
            # Assume get_obstacle_info returns a dictionary with keys such as:
            # "arm_view_score": how good the view is for observing the arm.
            # "nearest_obstacle_dist": the distance to the closest obstacle.
            obs_info = self.perception.get_obstacle_info(model=model, data=data)
            # Extract values with safe defaults.
            arm_view_score = obs_info.get("arm_view_score", 0.0)
            nearest_dist = obs_info.get("nearest_obstacle_dist", 1.0)  # assume ≥1.0 is safe

            # Reward good view score; here, we reuse a scale, but you can add a separate one.
            reward_perception += self._config.reward_config.scales.pose * arm_view_score

            # Penalize if too close to obstacles.
            if nearest_dist < 0.5:
                reward_perception += self._config.reward_config.scales.dof_pos_limits * (nearest_dist - 0.5)

        # --- Control Effort Penalty ---
        # Sum absolute actuator forces to penalize excessive control effort.
        total_torque = jp.sum(jp.abs(data.actuator_force))
        reward_torque = self._config.reward_config.scales.torques * total_torque

        # --- Stand-Still Penalty ---
        # Penalize if the robot's planar speed is too low (e.g., to avoid trivial stand-still solutions).
        stand_still_penalty = 0.0
        if jp.linalg.norm(local_linvel[:2]) < 0.01:
            stand_still_penalty = self._config.reward_config.scales.stand_still

        # --- Combine Reward Terms ---
        reward = reward_tracking + reward_orientation + reward_perception + reward_torque + stand_still_penalty

        return reward

    def _get_termination(self, data: mujoco.MjData) -> jax.Array:
        
        # Determine if the current episode should be terminated.
        
        # --- Fall Termination ---
        # Retrieve the robot's up vector (assumed that self.get_upvector(data) returns [u_x, u_y, u_z]).
        up_vec = self.get_upvector(data)
        # We define a fall if the z-component of the up vector is below 0.5.
        fall_termination = up_vec[-1] < 0.5  # Adjust the threshold when tuning.

        # --- Out-of-Bounds Termination ---
        # Obtain the robot's current center-of-mass position.
        com_pos = self.robot_go2.pc
        # Define the environment boundaries for the x and y coordinates.
        bounds_min = -6.0
        bounds_max = 6.0
        # Check if either the x or y position is outside the defined boundaries.
        out_of_bounds = jp.any((com_pos[:2] < bounds_min) | (com_pos[:2] > bounds_max))

        # --- Collision-Based Termination ---
        # If a collision detection method is provided, use it to determine critical collision termination.
        # The function collision.has_critical_collision(data) should return True if a critical collision has occurred.
        if hasattr(collision, "has_critical_collision"):
            collision_termination = collision.has_critical_collision(data)
        else:
            collision_termination = False

        # --- Combine Termination Conditions ---
        # If any one of the conditions is met, the episode should terminate.
        terminated = fall_termination or out_of_bounds or collision_termination

        # Return the termination flag as a JAX array.
        return jp.array(terminated)


    def has_critical_collision(data: mujoco.MjData) -> bool:
        # Checks for critical collisions by examining the contact penetration depth
        # between bodies in the simulation.

        # Define a threshold for what constitutes a severe penetration.
        # For example, any contact with a penetration deeper than 0.02 (i.e., contact.dist < -0.02)
        # is treated as a critical collision.
        CRITICAL_PENETRATION_THRESHOLD = -0.02

        # Iterate over all current contacts.
        for i in range(data.ncon):
            contact = data.contact[i]
            # The 'dist' attribute represents the distance between surfaces:
            # a negative value indicates penetration.
            if contact.dist < CRITICAL_PENETRATION_THRESHOLD:
                return True

        # No critical contacts were detected.
        return False


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
