
from Navigator import Navigator
import mujoco
import numpy as np

from arm_mujoco.src.Robot  import RobotGo2
from arm_mujoco.src.Arm  import Arm
from arm_mujoco.src.Perception import Perception
from Locomotion_Controller import Locomotion_Controller
import base as go2_base
# NavigationPolicy Class: Defines the navigation policy, TBD
#  Until now: it has Navigator() which decides randomly the vel cmds


class NavigationPolicy: #go2_base.Go2NavEnv
    def __init__(
        self,
        default_angles: np.ndarray,
        action_scale: float = 0.5,
        vel_scale_x: float = 1.5,
        vel_scale_y: float = 0.8,
        vel_scale_rot: float = 2 * np.pi, 
        t_last_cmd: float = 0.0,
        ctrl_dt: float = 0.01,
        sim_dt : float = 0.002, #model.opt.timestep
        _ONNX_DIR = None,
        model = None
       ):
        self.n_substeps = int(round(ctrl_dt / sim_dt))
        self._default_angles = default_angles
        self._action_scale = action_scale
        self._counter = 0
        self.t_last_cmd = 0.0
        self.dt_new_cmd = 0.68 # HERE TBD
        self.robot_go2 = RobotGo2()

        # Constructor Locomotion_Controller
        self.Loc_Ctrl = Locomotion_Controller(
        locomotion_policy_path=(_ONNX_DIR / "go2_policy_galloping.onnx").as_posix(),
        # Pass only qpos regarding Go2's joints, NOT base's x,y,z,quat 
        default_angles=np.array(model.keyframe("home").qpos[self.robot_go2.i_start_qpos:self.robot_go2.i_end_qpos]),
        # Pass different n_substeps if need. Works: int(round(0.01 / 0.002))
        n_substeps=self.n_substeps, 
        action_scale=0.5,
        vel_scale_x=1.5,
        vel_scale_y=0.8,
        vel_scale_rot=2 * np.pi,
        locomotion_cmd=np.zeros(3)) # Define which Navigator-Boss the locomotion cotroller will have.

        # Added this to decide the vel Cmds: in future, it will come from Navigation Policy
        self.Navigator_ = Navigator()
        # # Obj. under arm_mujoco
        self.arm = Arm()
        # self.perception = Perception()  #, perception_cont
            # Obj. for RL control
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



    def _get_obs(self):
        pass