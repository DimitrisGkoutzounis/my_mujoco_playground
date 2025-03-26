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
"""Defines Unitree Go2 quadruped constants."""

from etils import epath

from mujoco_playground._src import mjx_env
# from mujoco_playground._src.dynamic_events.arm_mujoco.src.Arm import Arm

ROOT_PATH = mjx_env.ROOT_PATH / "dynamic_events/arm_mujoco"
UR5E_GO2_SCENE = (
    ROOT_PATH / "xml" / "scene.xml"
)



def task_to_xml(task_name: str) -> epath.Path:
  return {
      "arm_go2_simple_terrain": UR5E_GO2_SCENE,
    #   "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }[task_name]


FEET_SITES = [
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot",
]

FEET_GEOMS = [
    "FR",
    "FL",
    "RR",
    "RL",
]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "base"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
