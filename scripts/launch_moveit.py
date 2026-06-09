"""
Grasp_fruit MoveIt 론치 파일 (planning only — 실행은 SHM bridge 사용)

실행 노드:
  1. move_group       — OMPL + pick_ik IK (/compute_ik, /compute_cartesian_path, /move_action)
  2. robot_state_publisher — TF 발행 (robot_description 기반)

전제:
  - franka_kistar_moveit_config 패키지가 소싱되어 있어야 함
  - franka_description (apt: ros-humble-franka-description) 설치됨
  - /joint_states 는 franka_joint_state_relay.py 가 발행
"""

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _load_yaml(pkg: str, rel: str) -> dict:
    path = os.path.join(get_package_share_directory(pkg), rel)
    with open(path) as f:
        return yaml.safe_load(f)


def _load_file(pkg: str, rel: str) -> str:
    path = os.path.join(get_package_share_directory(pkg), rel)
    with open(path) as f:
        return f.read()


def generate_launch_description():
    # ── URDF (franka_kistar_description, ros2_control 비활성) ──────────────────
    xacro_file = os.path.join(
        get_package_share_directory('franka_kistar_description'),
        'urdf', 'fr3_kistar.urdf.xacro',
    )
    robot_description_cmd = Command([
        FindExecutable(name='xacro'), ' ', xacro_file,
        ' ros2_control:=false',
        ' use_fake_hardware:=false',
        ' robot_ip:=172.16.0.1',
    ])
    robot_description = {
        'robot_description': ParameterValue(robot_description_cmd, value_type=str)
    }

    # ── SRDF (정적 파일 직접 로드) ─────────────────────────────────────────────
    srdf_str = _load_file('franka_kistar_moveit_config', 'config/fr3_kistar.srdf')
    robot_description_semantic = {'robot_description_semantic': srdf_str}

    # ── Kinematics (pick_ik) ───────────────────────────────────────────────────
    kinematics_yaml = _load_yaml('franka_kistar_moveit_config', 'config/kinematics.yaml')

    # ── Joint limits ───────────────────────────────────────────────────────────
    joint_limits_yaml = _load_yaml('franka_kistar_moveit_config', 'config/joint_limits.yaml')
    robot_description_planning = {'robot_description_planning': joint_limits_yaml}

    # ── OMPL planning pipeline ─────────────────────────────────────────────────
    ompl_yaml = _load_yaml('franka_kistar_moveit_config', 'config/ompl_planning.yaml')
    ompl_config = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters':
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/ResolveConstraintFrames '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl_config['move_group'].update(ompl_yaml)

    # ── Planning scene monitor ─────────────────────────────────────────────────
    planning_scene_monitor = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    # ── move_group 노드 ────────────────────────────────────────────────────────
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_planning,
            kinematics_yaml,
            ompl_config,
            planning_scene_monitor,
            {
                'trajectory_execution.allowed_execution_duration_scaling': 3.0,
                'trajectory_execution.allowed_goal_duration_margin': 2.0,
                'trajectory_execution.allowed_start_tolerance': 0.05,
                'moveit_manage_controllers': False,
            },
        ],
    )

    # ── robot_state_publisher (TF 발행) ───────────────────────────────────────
    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    return LaunchDescription([
        move_group_node,
        robot_state_pub_node,
    ])
