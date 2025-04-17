from utils import *
import warnings
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'


xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

xml_path = "/home/dimitris/Documents/Github/adapted_mujoco_playground/mujoco_playground/_src/dynamic_events/arm_mujoco/xml/scene.xml"
# xml_path = "/home/dimitris/Documents/Github/go2_rl_mjc/resources/unitree_go2/scene_mjx.xml"

# xml_path = "/home/dimitris/Documents/Github/adapted_mujoco_playground/mujoco_playground/_src/dm_control_suite/xmls/cartpole.xml"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Make model, data, and renderer
# mj_model = mujoco.MjModel.from_xml_string(xml)
mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

print(mj_data.qpos, type(mj_data.qpos))
print(mjx_data.qpos, type(mjx_data.qpos), mjx_data.qpos.devices())

def default_vision_config() -> config_dict.ConfigDict:
  return config_dict.create(
      gpu_id=0, #id of the gpu
      render_batch_size=512, #frames in a single batch
      render_width=32, #resulution of rendered images
      render_height=32, # 64x64
      enabled_geom_groups=[0, 1, 2], # XML geometric groups to render
      use_rasterizer=False, #rasterizer converts 3D to 2D
      history=1, #number of past frames to keep
  )


# def default_config() -> config_dict.ConfigDict:
#   return config_dict.create(
#       ctrl_dt=0.01,
#       sim_dt=0.01,
#       episode_length=1000,
#       action_repeat=1,
#       vision=False,
#       vision_config=default_vision_config(),
#   )


config = default_vision_config()
config.render_batch_size = 128



# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

jit_step = jax.jit(mjx.step)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)
while mjx_data.time < duration:
  mjx_data = jit_step(mjx_model, mjx_data)
  if len(frames) < mjx_data.time * framerate:
    mj_data = mjx.get_data(mj_model, mjx_data)
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)

try:
    # pylint: disable=import-outside-toplevel
    from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
    print("ok")
except ImportError:
    warnings.warn("Madrona MJX not installed. Cannot use vision with.")

renderer = BatchRenderer(
    m=mjx_model,
    gpu_id=config.gpu_id,
    num_worlds=config.render_batch_size,
    batch_render_view_width=config.render_width,
    batch_render_view_height=config.render_height,
    enabled_geom_groups=np.asarray(config.enabled_geom_groups),
    enabled_cameras=np.asarray([0]),
    add_cam_debug_geo=False,
    use_rasterizer=False,      # rasterizer on
    viz_gpu_hdls=None
)

print("Madrna batch renderer loaded succesfully")

# self.renderer = BatchRenderer(
#           m=self._mjx_model,
#           gpu_id=self._config.vision_config.gpu_id,
#           num_worlds=self._config.vision_config.render_batch_size,
#           batch_render_view_width=self._config.vision_config.render_width,
#           batch_render_view_height=self._config.vision_config.render_height,
#           enabled_geom_groups=np.asarray(
#               self._config.vision_config.enabled_geom_groups
#           ),
#           enabled_cameras=np.asarray([
#               0,
#           ]),
#           add_cam_debug_geo=False,
#           use_rasterizer=self._config.vision_config.use_rasterizer,
#           viz_gpu_hdls=None,
#       )

# Simulate and display video.
save_video(frames, fps=framerate)

rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, 256) #was 4096
batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (1,))))(rng)

jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch = jit_step(mjx_model, batch)

print(batch.qpos)

batched_mj_data = mjx.get_data(mj_model, batch)
print([d.qpos for d in batched_mj_data])