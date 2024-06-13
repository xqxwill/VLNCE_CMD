import logging


class RobotMove:
    def __init__(self, client):
        self.client = client
        self.action_mapping = {
            0: 'STOP',
            1: 'MOVE_FORWARD',
            2: 'TURN_LEFT',
            3: 'TURN_RIGHT'
        }

    def exec_cmd(self, cmd):
        """Execute a command on the remote client."""
        try:
            stdin, stdout, stderr = self.client.exec_command(cmd)
            stdout_output = stdout.read().decode()
            stderr_output = stderr.read().decode()
            if stdout_output:
                logging.info(stdout_output)
            if stderr_output:
                logging.error(stderr_output)
            return stdout_output, stderr_output
        except Exception as e:
            logging.error(f"Failed to execute command: {cmd}\nError: {e}")
            return None, str(e)

    def move(self, action):
        """Send a movement command to the robot."""
        action_name = self.action_mapping.get(action.item(), None)
        if action_name:
            cmd = self._get_command_for_action(action_name)
            if cmd:
                self.exec_cmd(cmd)
            else:
                logging.warning(f"Unknown action name: {action_name}")
        else:
            logging.warning(f"Unknown action: {action}")

    def _get_command_for_action(self, action):
        """Get the command corresponding to the action."""
        commands = {
            'MOVE_FORWARD': 'bash --login -c "ros2 service call /mi_desktop_48_b0_2d_5f_bf_4b/motion_result_cmd protocol/srv/MotionResultCmd \'{motion_id: 303, vel_des:[-0.25, 0, 0], step_height: [0.05, 0.05], duration: 1000}\'"',
            'TURN_LEFT': 'bash --login -c "ros2 service call /mi_desktop_48_b0_2d_5f_bf_4b/motion_result_cmd protocol/srv/MotionResultCmd \'{motion_id: 303, vel_des:[0, 0, 1.5708], step_height: [0.05, 0.05], duration: 1000}\'"',
            'TURN_RIGHT': 'bash --login -c "ros2 service call /mi_desktop_48_b0_2d_5f_bf_4b/motion_result_cmd protocol/srv/MotionResultCmd \'{motion_id: 303, vel_des:[0, 0, -1.5708], step_height: [0.05, 0.05], duration: 1000}\'"',
            'STOP': 'bash --login -c "ros2 service call /mi_desktop_48_b0_2d_5f_bf_4b/motion_result_cmd protocol/srv/MotionResultCmd \'{motion_id: 303, vel_des:[0, 0, 0], step_height: [0.05, 0.05], duration: 100}\'"',
        }
        return commands.get(action)
