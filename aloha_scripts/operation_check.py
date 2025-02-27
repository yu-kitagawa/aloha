import time
import sys
import IPython
e = IPython.embed

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from constants import MASTER2PUPPET_JOINT_FN, DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE
from robot_utils import torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions

def prep_robots(master_bot, puppet_bot):
    # reboot gripper motors, and set operating modes for all motors
    puppet_bot.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    puppet_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    # master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    # master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit
    torque_on(puppet_bot)
    # torque_on(master_bot)

    # move arms to starting position
    start_arm_qpos = START_ARM_POSE[:6]
    # move_arms([master_bot, puppet_bot], [start_arm_qpos] * 2, move_time=1)
    move_arms([puppet_bot], [start_arm_qpos] * 1, move_time=1)

    # move grippers to starting position
    # move_grippers([master_bot, puppet_bot], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)
    move_grippers([puppet_bot], [PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)

def teleop(robot_side):
    """ A standalone function for experimenting with teleoperation. No data recording. """
    puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', init_node=True)
    # master_bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_{robot_side}', init_node=False)
    master_bot = None

    prep_robots(master_bot, puppet_bot)
    # press_to_start(master_bot)


    ## --
    print ("Move sleep pose")

    puppet_sleep_position = (0, -1.7, 1.55, 0.12, 0.65, 0)
    # master_sleep_position = (0, -1.1, 1.24, 0, -0.24, 0)
    move_arms([puppet_bot], [puppet_sleep_position] * 1, move_time=2)

if __name__=='__main__':
    side = sys.argv[1]
    teleop(side)
