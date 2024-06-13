# This is the scene we are going to load.
# we support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
# 加载场景文件
import os

import habitat_sim
import numpy as np
from rich.jupyter import display

from examples.tutorials.nb_python.Habitat_Interactive_Tasks import sim, \
    display_sample

test_scene = os.path.join(
    "/home/x/VLNCE/VLN-CE/data/", "scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
)
# 指定默认的Agent和几个传感器参数，如与Agent的相对位置
sim_settings = {
    "scene": test_scene,  # Scene path，场景路径
    "default_agent": 0,  # Index of the default agent，默认的Agent索引值
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent，传感器参数与Agent的相对位置
    "width": 256,  # Spatial resolution of the observations，观测的分辨率
    "height": 256,
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
# 配置包括两部分：模拟器后端和Agent；
def make_simple_cfg(settings):
    # simulator backend，模拟器后端配置
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent，Agent配置
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    # 可以将多个传感器加到Agent上，这里加了一个RGBD相机
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor" # 名称
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR # 传感器类型
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]] # 分辨率
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0] # 相对位置

    agent_cfg.sensor_specifications = [rgb_sensor_spec] # 通过列表的形式设定传感器

    return habitat_sim.Configuration(sim_cfg, [agent_cfg]) # 返回模拟器的配置


cfg = make_simple_cfg(sim_settings)
try:  # Needed to handle out of order cell run in Jupyter
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)
# initialize an agent，初始化Agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state，设定Agent的状态
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space，设定Agent在世界中的初始位置
agent.set_state(agent_state)

# Get agent state，获取Agent的状态
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation) # 打印Agent当前位置以及四元数的旋转

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
# 定义了包含三个动作的离散动作空间：前进、左转和右转，可以自定义离散动作空间自定义动作
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        if display:
            display_sample(observations["color_sensor"])


action = "turn_right"
navigateAndSee(action)

action = "turn_right"
navigateAndSee(action)

action = "move_forward"
navigateAndSee(action)

action = "turn_left"
navigateAndSee(action)

# action = "move_backward"   // #illegal, no such action in the default action space，非法动作空间中无该动作
# navigateAndSee(action)
