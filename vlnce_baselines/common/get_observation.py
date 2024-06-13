import os
import cv2
import numpy as np

class LocalImageEnv:
    def __init__(self, image_folder_path):
        self.image_folder_path = image_folder_path
        self.image_files = sorted(os.listdir(image_folder_path))
        self.current_image_index = 0

    def reset(self):
        self.current_image_index = 0
        return self._get_current_image()

    def step(self, action):
        # In a real environment, you might have different observations and rewards based on actions
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        observation = self._get_current_image()
        reward = 0  # No reward in this simple example
        done = False  # We're never 'done' in this simple example
        info = {}  # Additional information, unused in this simple example
        return observation, reward, done, info

    def _get_current_image(self):
        image_path = os.path.join(self.image_folder_path, self.image_files[self.current_image_index])
        image = cv2.imread(image_path)
        return image

class VectorLocalImageEnv:
    def __init__(self, image_folder_path, num_envs):
        self.envs = [LocalImageEnv(image_folder_path) for _ in range(num_envs)]
        self.num_envs = num_envs

    def reset(self):
        return [env.reset() for env in self.envs]

    def step(self, actions):
        return [env.step(action) for env, action in zip(self.envs, actions)]

    def render(self, mode='human'):
        images = [env._get_current_image() for env in self.envs]
        tile = tile_images(images)
        if mode == "human":
            cv2.imshow("vecenv", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    def close(self):
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
