import time
import os
import paramiko
import cv2
import json
import numpy as np
from gym import spaces
from habitat.core.spaces import ActionSpace, EmptySpace
from vlnce_baselines.common.robot_connection import RobotMove
from transformers import pipeline
from PIL import Image
import logging
import torch


class RealEnv:
    def __init__(self, client):
        self.instruction_json_path = "/home/x/VLNCE/TEST/instruction.json"
        self.rgb_image_path = "/home/x/VLNCE/TEST/image/RGB_image.jpg"
        self.depth_image_path = "/home/x/VLNCE/TEST/image/Depth_image.jpg"
        self.depth_image_path_directory = "/home/x/VLNCE/TEST/image"
        self.rgb_shape = (256, 256, 3)
        self.depth_shape = (256, 256, 1)
        self.remote_folder = "/home/unitree/Camera"
        self.local_folder = "/home/x/VLNCE/TEST/image"
        self.client = client
        self.robot_move = RobotMove(client)
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=self.rgb_shape,
                              dtype=np.uint8),
            'depth': spaces.Box(low=0, high=255, shape=self.depth_shape,
                                dtype=np.float32),
            'instruction': spaces.MultiDiscrete([256]),
            # Adjusted to match expected instruction format
        })
        self.action_space = ActionSpace({
            "MOVE_FORWARD": EmptySpace(),
            "STOP": EmptySpace(),
            "TURN_LEFT": EmptySpace(),
            "TURN_RIGHT": EmptySpace()
        })
        self.num_envs = 1
        self.current_episode = None
        self.reset()

    def download_folder(self):
        """
        将机器狗采集的图像传到服务器端
        Returns:

        """
        sftp = self.client.open_sftp()
        try:
            for filename in sftp.listdir(self.remote_folder):
                if filename.endswith('.jpg'):
                    remote_filepath = os.path.join(self.remote_folder,
                                                   filename)
                    local_filepath = os.path.join(self.local_folder,
                                                  'RGB_image.jpg')
                    sftp.get(remote_filepath, local_filepath)
            logging.info("Images downloaded successfully.")
        except Exception as e:
            logging.error(f"Error downloading images: {e}")
        finally:
            sftp.close()

    def take_picture(self):
        try:
            # delete_image_cmd = 'mv /home/mi/Camera/*.jpg'
            move_image_cmd = 'mv /home/unitree/Camera/*.jpg /home/unitree/old'
            self.robot_move.exec_cmd(move_image_cmd)  # Use instance method
            take_picture_cmd = (
                'bash --login -c "ros2 service call /mi_desktop_48_b0_2d_5f_bf_4b/camera_service protocol/srv/CameraService \'{command: 1 , args: \\"width=256;height=256\\", width: 256, height: 256, fps: 30}\'"')
            self.robot_move.exec_cmd(take_picture_cmd)  # Use instance method
            self.download_folder()
        except Exception as e:
            logging.error(f"Error taking picture: {e}")

    def get_depth(self):

        rgb_image = Image.open(self.rgb_image_path)

        pipe = pipeline(task="depth-estimation",
                        model="LiheYoung/depth-anything-base-hf")
        depth = pipe(rgb_image)["depth"]
        depth_np = np.array(depth, dtype=np.float32)
        inv_depth = (depth_np - 255) * -1
        depth_new = inv_depth.reshape(inv_depth.shape[0], inv_depth.shape[1],
                                      1).astype(np.float32)
        print(depth_new.shape)
        # #cv2.imshow('Image2', depth_np)
        # #cv2.waitKey(0)
        # #cv2.destroyAllWindows()
        # # Normalize the depth image to the range [0, 1]
        min_depth = np.min(depth_new)
        max_depth = np.max(depth_new)
        depth_normalized = (depth_new - min_depth) / (max_depth - min_depth)

        # # Ensure values are within [0, 1] range
        depth_normalized = np.clip(depth_normalized, 0, 1)

        # # Convert the normalized depth image back to a numpy array if needed
        depth_nor = (depth_normalized * 255).astype(np.uint8)

        cv2.imshow('Normalized Depth Image', depth_nor)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        filename = "Depth_image.png"

        cv2.imwrite(os.path.join(self.depth_image_path_directory, filename),
                    depth_nor)

    def reset(self):
        with open(self.instruction_json_path, 'r') as f:
            instruction_data = json.load(f)
        self.take_picture()
        self.get_depth()
        self.rgb_image = cv2.imread(self.rgb_image_path)
        self.depth_image_n = cv2.imread(self.depth_image_path,
                                        cv2.IMREAD_GRAYSCALE)
        self.depth_image = (self.depth_image_n / 255).astype(np.float32)
        self.instruction = instruction_data
        self.current_episode = instruction_data['episode_id']
        self.step_count = 0
        return self._get_observation()

    def _get_observation(self):

        rgb_observation = self.rgb_image
        depth_observation = self.depth_image
        # Resize depth observation to the expected size (assuming 256x256 here, change if necessary)
        # Ensure it has a batch dimension and the correct shape
        # depth_observation = np.expand_dims(depth_observation, axis=0)
        if depth_observation.shape != (
        self.depth_shape[0], self.depth_shape[1]):
            depth_observation = cv2.resize(depth_observation, (
            self.depth_shape[1], self.depth_shape[0]))
        depth_observation = depth_observation.reshape(self.depth_shape)
        depth_observation = torch.tensor(depth_observation,
                                         dtype=torch.float32)  # Shape: (1, 256, 256, 1)
        rgb_observation = cv2.resize(rgb_observation,
                                     (self.rgb_shape[1], self.rgb_shape[0]))
        rgb_observation = torch.tensor(rgb_observation, dtype=torch.uint8)
        # Convert depth observation to torch tensor and add batch dimension
        # depth_observation = torch.tensor(depth_observation).unsqueeze(0)
        instruction = {
            'instruction_text': self.instruction['instruction'][
                'instruction_text'],
            'tokens': self.instruction['instruction']['instruction_tokens']
        }
        print(f"Depth observation shape: {depth_observation.shape}")
        print(f"RGB observation shape: {rgb_observation.shape}")

        return [{
            'rgb': rgb_observation,
            'depth': depth_observation,
            'instruction': instruction
            # Include the flattened instruction data
        }]

    def current_episodes(self):
        return [self.current_episode]

    def step(self, action):

        self.robot_move.move(action)
        # Wait for a short period for the robot to stabilize or reach the new position
        time.sleep(2)
        # Take a new picture and get the depth
        self.take_picture()
        self.get_depth()
        # Update the rgb_image and depth_image variables with the newly captured images
        self.rgb_image = cv2.imread(self.rgb_image_path)
        self.depth_image = cv2.imread(self.depth_image_path,
                                      cv2.IMREAD_GRAYSCALE)
        # Update the observation with the new images
        self.step_count += 1
        return self._get_observation()

    def get_done(self, actions):
        done = False
        self.action_name = actions
        if self.action_name == "STOP":
            done = True
        return [done]  # Ensure this is a list

    def close(self):
        try:
            self.client.close()
            logging.info("SSH client closed successfully.")
        except Exception as e:
            logging.error(f"Error closing SSH client: {e}")
