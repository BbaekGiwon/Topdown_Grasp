#!/usr/bin/env python3
"""
GraspExecutor + frame-transform helpers (extracted from robot_executor.py).

Import this module instead of robot_executor when you only need the class:
    from utils.grasp import GraspExecutor, world_to_base
"""

import math
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath, GetPositionIK, GetPositionFK
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Float32MultiArray, Float64, Float64MultiArray, Int16MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration as RosDuration

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from utils.step import (
    step_approach, step_descend, step_lift,
    step_init_hand, step_close_hand, step_release_hand, step_go_home,
)
from utils.hand import HAND_INIT_ENC, HAND_RELEASE_ENC, HAND_STEPS, HAND_PERIOD
from utils.arm import (
    PLANNING_GROUP, EE_LINK, REF_FRAME, PLANNING_TIME,
    HOME_JOINT_NAMES, HOME_JOINT_VALUES,
)

TRAJ_TOPIC = '/franka/target_trajectory'
DEG_TO_RAW = 8192.0 / 180.0


# ── Frame transform helpers ───────────────────────────────────────────────────

def _quat_to_rotmat(xyzw):
    x, y, z, w = xyzw
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


def _rotmat_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return [float((R[2,1]-R[1,2])*s), float((R[0,2]-R[2,0])*s),
                float((R[1,0]-R[0,1])*s), float(0.25/s)]
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return [float(0.25*s), float((R[0,1]+R[1,0])/s),
                float((R[0,2]+R[2,0])/s), float((R[2,1]-R[1,2])/s)]
    if R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return [float((R[0,1]+R[1,0])/s), float(0.25*s),
                float((R[1,2]+R[2,1])/s), float((R[0,2]-R[2,0])/s)]
    s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
    return [float((R[0,2]+R[2,0])/s), float((R[1,2]+R[2,1])/s),
            float(0.25*s), float((R[1,0]-R[0,1])/s)]


def world_to_base(T_world_base_list, xyz_world, quat_world_xyzw):
    """Transform EE pose from world frame to base frame.

    T_world_base: 4x4 list (world<-base, i.e. p_world = T @ p_base).
    Returns (xyz_base, quat_base_xyzw).
    """
    T_wb = np.array(T_world_base_list, dtype=np.float64)
    R_wb = T_wb[:3, :3]
    t_wb = T_wb[:3, 3]
    R_bw = R_wb.T
    t_bw = -R_bw @ t_wb
    p_b  = R_bw @ np.array(xyz_world, dtype=np.float64) + t_bw
    R_we = _quat_to_rotmat(quat_world_xyzw)
    R_be = R_bw @ R_we
    q_b  = _rotmat_to_quat(R_be)
    return p_b.tolist(), q_b


# ── Node ─────────────────────────────────────────────────────────────────────

class GraspExecutor(Node):

    def __init__(self, summary: dict, execute_mode: str, speed_factor: float,
                 approach_offset: float, summary_json_path: str = ''):
        super().__init__('grasp_executor')
        self._summary           = summary
        self._mode              = execute_mode
        self._speed             = speed_factor
        self._approach_offset   = summary.get('approach_offset', approach_offset)
        self._summary_json_path = summary_json_path
        self._hand_deg          = None
        self._last_hand_enc     = None
        self._success           = False
        self._current_joints    = None
        self._approach_traj     = None
        self._cb_group          = ReentrantCallbackGroup()
        self._setup_ros()
        self._thread = threading.Thread(target=self._run_guarded, daemon=True)
        self._thread.start()

    # ── ROS setup ─────────────────────────────────────────────────────────────

    def _setup_ros(self):
        self._mg_client   = ActionClient(self, MoveGroup, '/move_action',
                                         callback_group=self._cb_group)
        self._cart_client = self.create_client(GetCartesianPath,
                                               '/compute_cartesian_path',
                                               callback_group=self._cb_group)
        self._ik_client   = self.create_client(GetPositionIK, '/compute_ik',
                                               callback_group=self._cb_group)
        self._display_pub = self.create_publisher(
            DisplayTrajectory, '/display_planned_path', 10)
        self._hand_pub = self.create_publisher(
            Int16MultiArray, '/hand/target_joint', 10)
        _hand_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Float32MultiArray, '/hand/joint_position',
                                 self._hand_cb, _hand_qos)
        self.create_subscription(Float64MultiArray, '/franka/joint_position',
                                 self._franka_joint_cb, 10)
        self._franka_target_pub = self.create_publisher(
            Float64MultiArray, '/franka/target_joint', 10)
        self._franka_speed_pub  = self.create_publisher(
            Float64, '/franka/target_speed_factor', 10)
        self._traj_smooth_pub = self.create_publisher(
            JointTrajectory, TRAJ_TOPIC, 1)
        self._traj_client = None

        self.get_logger().info('Waiting for MoveGroup action server...')
        if not self._mg_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(
                'MoveGroup 서버를 찾을 수 없습니다 (15 s timeout).\n'
                '  → 컨테이너 내에서 MoveIt이 실행 중인지 확인하세요:\n'
                '    start_robot_gwb.sh 를 먼저 실행한 뒤 다시 시도하세요.')
            rclpy.shutdown(); sys.exit(1)
        self.get_logger().info('Waiting for /compute_cartesian_path service...')
        if not self._cart_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error(
                '/compute_cartesian_path 서비스를 찾을 수 없습니다 (15 s timeout).')
            rclpy.shutdown(); sys.exit(1)
        self.get_logger().info('Waiting for /compute_ik service...')
        if not self._ik_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error('/compute_ik 서비스를 찾을 수 없습니다 (15 s timeout).')
            rclpy.shutdown(); sys.exit(1)
        self._fk_client = self.create_client(GetPositionFK, '/compute_fk',
                                             callback_group=self._cb_group)
        self.get_logger().info('Waiting for /compute_fk service...')
        if not self._fk_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error('/compute_fk 서비스를 찾을 수 없습니다 (15 s timeout).')
            rclpy.shutdown(); sys.exit(1)
        self.get_logger().info(f'Servers ready  (mode={self._mode})')

    def _hand_cb(self, msg: Float32MultiArray):
        if len(msg.data) == 16:
            self._hand_deg = list(msg.data)

    def _franka_joint_cb(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self._current_joints = list(msg.data)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _make_pose(self, x, y, z, qx, qy, qz, qw) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id  = REF_FRAME
        p.header.stamp     = self.get_clock().now().to_msg()
        p.pose.position    = Point(x=float(x), y=float(y), z=float(z))
        p.pose.orientation = Quaternion(x=float(qx), y=float(qy),
                                        z=float(qz), w=float(qw))
        return p

    def _wait(self, future, timeout: float) -> bool:
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.01)
        return True

    def _confirm(self, prompt: str) -> bool:
        while True:
            try:
                resp = input(prompt).strip().lower()
            except EOFError:
                return False
            if resp == 'y':
                return True
            if resp == 'n':
                return False
            print("  'y' 또는 'n' 을 입력하세요.")

    def _display_trajectory(self, start_state, trajectory, label: str):
        disp = DisplayTrajectory()
        disp.model_id = PLANNING_GROUP
        disp.trajectory_start = start_state
        disp.trajectory.append(trajectory)
        time.sleep(0.1)
        self._display_pub.publish(disp)
        time.sleep(0.3)
        self.get_logger().info(f'[{label}] RViz에 경로 표시됨 → 확인 후 y/n 입력')

    # ── Planning ──────────────────────────────────────────────────────────────

    _JOINT_VEL_LIMITS = [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]

    def _add_timestamps(self, jt: JointTrajectory) -> JointTrajectory:
        pts = list(jt.points)
        if len(pts) < 2:
            return jt
        t     = 0.0
        speed = max(0.001, self._speed)
        pts[0].time_from_start = RosDuration(sec=0, nanosec=0)
        for i in range(1, len(pts)):
            q_prev = list(pts[i-1].positions)
            q_curr = list(pts[i].positions)
            dt = max(
                abs(q_curr[j] - q_prev[j]) / (self._JOINT_VEL_LIMITS[j] * speed)
                for j in range(min(7, len(q_curr)))
            )
            t += max(dt, 1e-3)
            sec  = int(t)
            nsec = int((t - sec) * 1e9)
            pts[i].time_from_start = RosDuration(sec=sec, nanosec=nsec)
        result = JointTrajectory()
        result.joint_names = list(jt.joint_names) if jt.joint_names else list(HOME_JOINT_NAMES)
        result.points = pts
        self.get_logger().info(f'[TIMESTAMP] {len(pts)} pts  total={t:.2f}s')
        return result

    def _plan_cartesian(self, goal: PoseStamped, label: str,
                        jnames=None, jvals=None, max_step: float = 0.005):
        req = GetCartesianPath.Request()
        req.header           = goal.header
        req.group_name       = PLANNING_GROUP
        req.link_name        = EE_LINK
        req.waypoints        = [goal.pose]
        req.max_step         = max_step
        req.jump_threshold   = 0.0
        req.avoid_collisions = True
        if jnames is not None:
            req.start_state.is_diff = False
            req.start_state.joint_state.name     = list(jnames)
            req.start_state.joint_state.position = list(jvals)
        else:
            req.start_state.is_diff = True
        self.get_logger().info(
            f'[{label}] Cartesian → '
            f'({goal.pose.position.x:.3f}, {goal.pose.position.y:.3f}, '
            f'{goal.pose.position.z:.3f})')
        future = self._cart_client.call_async(req)
        if not self._wait(future, 12.0):
            self.get_logger().error(f'[{label}] Cartesian timeout')
            return None
        res = future.result()
        if res.fraction < 0.9:
            self.get_logger().warning(
                f'[{label}] Cartesian {res.fraction:.0%} < 90% — 경로 계획 불완전')
            return None
        n = len(res.solution.joint_trajectory.points)
        self.get_logger().info(f'[{label}] Cartesian OK {res.fraction:.0%}  {n} pts')
        return res

    def _plan_ompl(self, goal: PoseStamped, label: str):
        g = MoveGroup.Goal()
        g.request.group_name                      = PLANNING_GROUP
        g.request.num_planning_attempts           = 20
        g.request.allowed_planning_time           = PLANNING_TIME
        g.request.max_velocity_scaling_factor     = 0.5
        g.request.max_acceleration_scaling_factor = 0.5
        ws = g.request.workspace_parameters
        ws.header.frame_id = REF_FRAME
        ws.min_corner.x = ws.min_corner.y = ws.min_corner.z = -5.0
        ws.max_corner.x = ws.max_corner.y = ws.max_corner.z =  5.0
        g.request.start_state.is_diff = True
        gc  = Constraints()
        pc  = PositionConstraint()
        pc.header            = goal.header
        pc.link_name         = EE_LINK
        pc.constraint_region = BoundingVolume()
        sph = SolidPrimitive()
        sph.type             = SolidPrimitive.SPHERE
        sph.dimensions       = [0.005]
        pc.constraint_region.primitives      = [sph]
        pc.constraint_region.primitive_poses = [goal.pose]
        pc.weight            = 1.0
        oc  = OrientationConstraint()
        oc.header            = goal.header
        oc.link_name         = EE_LINK
        oc.orientation       = goal.pose.orientation
        oc.absolute_x_axis_tolerance = 0.05
        oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        oc.weight            = 1.0
        gc.position_constraints    = [pc]
        gc.orientation_constraints = [oc]
        g.request.goal_constraints = [gc]
        g.planning_options.plan_only = True
        self.get_logger().info(
            f'[{label}] OMPL → '
            f'({goal.pose.position.x:.3f}, {goal.pose.position.y:.3f}, '
            f'{goal.pose.position.z:.3f})')
        future  = self._mg_client.send_goal_async(g)
        timeout = PLANNING_TIME + 4.0
        if not self._wait(future, timeout):
            self.get_logger().error(f'[{label}] OMPL goal timeout'); return None
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error(f'[{label}] OMPL goal rejected'); return None
        rf = gh.get_result_async()
        if not self._wait(rf, timeout):
            self.get_logger().error(f'[{label}] OMPL result timeout'); return None
        res = rf.result().result
        if res.error_code.val != 1:
            self.get_logger().error(f'[{label}] OMPL failed  code={res.error_code.val}')
            return None
        n = len(res.planned_trajectory.joint_trajectory.points)
        self.get_logger().info(f'[{label}] OMPL OK  {n} pts')
        return res

    def _plan_step(self, goal: PoseStamped, label: str,
                   jnames=None, jvals=None, confirm: bool = True):
        """Cartesian -> seeded IK+OMPL -> seeded IK+PtoP 순서로 시도."""
        seed         = list(jvals) if jvals is not None else self._current_joints
        jstart_names = HOME_JOINT_NAMES if seed is not None else None
        jstart_vals  = seed

        for max_step in [0.01, 0.02, 0.05, 0.1]:
            print(f'\n[{label}] Cartesian path (max_step={max_step:.3f})...')
            cart = self._plan_cartesian(goal, label,
                                        jnames=jstart_names, jvals=jstart_vals,
                                        max_step=max_step)
            if cart is not None:
                jt    = self._add_timestamps(cart.solution.joint_trajectory)
                frac  = cart.fraction
                partial_final = list(jt.points[-1].positions)
                completion_jt = None
                if frac < 1.0 and seed is not None:
                    self.get_logger().info(
                        f'[{label}] fraction={frac:.0%} → IK 보정 시도...')
                    j_comp = self._compute_ik_seeded(goal, partial_final, label + '_COMP')
                    if j_comp is not None:
                        completion_jt = self._make_joint_traj(j_comp, start=partial_final)
                        self.get_logger().info(f'[{label}] 보정 trajectory 생성 완료')
                self._display_trajectory(cart.start_state, cart.solution, label)
                if confirm and not self._confirm(f'  [{label}] 실행하시겠습니까? (y/n): '):
                    print(f'  [{label}] 취소됨.')
                    return None
                if completion_jt is not None:
                    self._exec_traj_smooth(jt)
                    dur = (jt.points[-1].time_from_start.sec +
                           jt.points[-1].time_from_start.nanosec * 1e-9)
                    self._wait_for_traj(dur)
                    return completion_jt
                return jt
            print(f'  fraction 부족 — 재시도 (max_step 완화)...')

        self.get_logger().warning(f'[{label}] Cartesian 전부 실패 → OMPL 시도...')
        if seed is not None:
            joints_target = self._compute_ik_seeded(goal, seed, label)
            if joints_target is not None:
                res = self._plan_joints(HOME_JOINT_NAMES, joints_target, label)
                if res is not None:
                    self._display_trajectory(
                        res.trajectory_start, res.planned_trajectory, label)
                    if confirm and not self._confirm(
                            f'  [{label}(OMPL)] 실행하시겠습니까? (y/n): '):
                        print(f'  [{label}] 취소됨.')
                        return None
                    return res.planned_trajectory.joint_trajectory
                self.get_logger().warning(f'[{label}] OMPL 실패 → direct PtoP...')
                delta = [abs(j - s) for j, s in zip(joints_target, seed)]
                print(f'  IK OK  max_Δ={math.degrees(max(delta)):.1f}°')
                if confirm and not self._confirm(
                        f'  [{label}(PtoP)] 실행하시겠습니까? (y/n): '):
                    print(f'  [{label}] 취소됨.')
                    return None
                return self._make_joint_traj(joints_target, start=seed)

        self.get_logger().error(f'[{label}] 경로 계획 실패. 중단.')
        return None

    def _plan_joints(self, joint_names, joint_values, label: str):
        g = MoveGroup.Goal()
        g.request.group_name                      = PLANNING_GROUP
        g.request.num_planning_attempts           = 10
        g.request.allowed_planning_time           = PLANNING_TIME
        g.request.max_velocity_scaling_factor     = 0.3
        g.request.max_acceleration_scaling_factor = 0.3
        if self._current_joints is not None:
            g.request.start_state.is_diff = False
            g.request.start_state.joint_state.name     = list(HOME_JOINT_NAMES)
            g.request.start_state.joint_state.position = [float(v) for v in self._current_joints]
        else:
            g.request.start_state.is_diff = True
        gc = Constraints()
        for name, val in zip(joint_names, joint_values):
            jc               = JointConstraint()
            jc.joint_name    = name
            jc.position      = float(val)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight        = 1.0
            gc.joint_constraints.append(jc)
        g.request.goal_constraints = [gc]
        g.planning_options.plan_only = True
        self.get_logger().info(f'[{label}] Joint-space planning...')
        future  = self._mg_client.send_goal_async(g)
        timeout = PLANNING_TIME + 4.0
        if not self._wait(future, timeout):
            self.get_logger().error(f'[{label}] goal timeout'); return None
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error(f'[{label}] goal rejected'); return None
        rf = gh.get_result_async()
        if not self._wait(rf, timeout):
            self.get_logger().error(f'[{label}] result timeout'); return None
        res = rf.result().result
        if res.error_code.val != 1:
            self.get_logger().error(f'[{label}] failed  code={res.error_code.val}')
            return None
        n = len(res.planned_trajectory.joint_trajectory.points)
        self.get_logger().info(f'[{label}] OK  {n} pts')
        return res

    def _compute_ik_seeded(self, goal: PoseStamped, seed_joints: list,
                           label: str) -> list | None:
        req = GetPositionIK.Request()
        req.ik_request.group_name       = PLANNING_GROUP
        req.ik_request.ik_link_name     = EE_LINK
        req.ik_request.pose_stamped     = goal
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec      = 1
        req.ik_request.timeout.nanosec  = 0
        req.ik_request.robot_state.joint_state.name     = HOME_JOINT_NAMES
        req.ik_request.robot_state.joint_state.position = [float(v) for v in seed_joints]
        req.ik_request.robot_state.is_diff              = False
        future = self._ik_client.call_async(req)
        if not self._wait(future, 5.0):
            self.get_logger().error(f'[{label}] IK timeout'); return None
        res = future.result()
        if res.error_code.val != 1:
            self.get_logger().warning(f'[{label}] IK failed  code={res.error_code.val}')
            return None
        js        = res.solution.joint_state
        joint_map = dict(zip(js.name, js.position))
        joints    = [joint_map[n] for n in HOME_JOINT_NAMES]
        delta     = [abs(j - s) for j, s in zip(joints, seed_joints)]
        self.get_logger().info(
            f'[{label}] IK OK  max_Δ={max(delta):.3f} rad  '
            f'joints={[round(v,4) for v in joints]}')
        return joints

    def _compute_fk(self, joint_values: list,
                    link_name: str = EE_LINK) -> list | None:
        req = GetPositionFK.Request()
        req.header.frame_id   = REF_FRAME
        req.fk_link_names     = [link_name]
        req.robot_state.joint_state.name     = list(HOME_JOINT_NAMES)
        req.robot_state.joint_state.position = [float(v) for v in joint_values]
        future = self._fk_client.call_async(req)
        if not self._wait(future, 5.0):
            self.get_logger().error('[FK] timeout'); return None
        res = future.result()
        if res.error_code.val != 1:
            self.get_logger().error(f'[FK] failed code={res.error_code.val}')
            return None
        p = res.pose_stamped[0].pose
        return [p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y,
                p.orientation.z, p.orientation.w]

    def _plan_step_ik(self, goal: PoseStamped, label: str,
                      seed_joints: list = None) -> list | None:
        seed = seed_joints if seed_joints is not None else self._current_joints
        if seed is not None:
            print(f'\n[{label}] Seeded IK (seed from current joints)...')
            joints = self._compute_ik_seeded(goal, seed, label)
            if joints is not None:
                delta = [abs(j - s) for j, s in zip(joints, seed)]
                print(f'  max_Δjoint = {max(delta):.3f} rad  '
                      f'({math.degrees(max(delta)):.1f}°)')
                if not self._confirm(f'  [{label}] 실행하시겠습니까? (y/n): '):
                    print(f'  [{label}] 취소됨.')
                    return None
                return joints
            self.get_logger().warning(f'[{label}] Seeded IK 실패 → Cartesian/OMPL fallback')
        jnames_seed = HOME_JOINT_NAMES if seed is not None else None
        jt = self._plan_step(goal, label, jnames=jnames_seed, jvals=seed)
        if jt is None:
            return None
        return list(jt.points[-1].positions)

    def _exec_joints(self, joint_values: list) -> None:
        spd      = Float64()
        spd.data = max(0.001, min(1.0, float(self._speed)))
        self._franka_speed_pub.publish(spd)
        time.sleep(0.05)
        tgt      = Float64MultiArray()
        tgt.data = [float(v) for v in joint_values]
        self._franka_target_pub.publish(tgt)
        self.get_logger().info(f'Joint cmd sent: {[round(v,4) for v in tgt.data]}')

    def _wait_for_traj(self, dur: float, margin: float = 0.3) -> None:
        wait = max(0.1, dur + margin)
        self.get_logger().info(f'[WAIT] {wait:.2f}s (dur={dur:.2f}s)')
        time.sleep(wait)

    def _wait_for_motion(self, target_joints: list,
                         timeout: float = 30.0, tol: float = 0.05) -> bool:
        time.sleep(0.1)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._current_joints is not None:
                err = max(abs(c - t)
                          for c, t in zip(self._current_joints, target_joints))
                if err < tol:
                    self.get_logger().info(
                        f'[WAIT] 도달 (err={err:.4f} rad, t={time.time()-t0:.1f}s)')
                    return True
            time.sleep(0.05)
        self.get_logger().warning(f'[WAIT] {timeout:.0f}s 초과')
        return False

    def _hold_hand_position(self, duration: float = 2.0) -> None:
        if not rclpy.ok():
            return
        enc = self._last_hand_enc
        if enc is None:
            return
        msg = Int16MultiArray()
        msg.data = [int(v) for v in enc]
        rate     = 30
        interval = 1.0 / rate
        count    = int(duration * rate)
        self.get_logger().info(f'[HOLD] 손 현재 위치 유지 ({duration:.1f}s)...')
        for _ in range(count):
            if not rclpy.ok():
                break
            self._hand_pub.publish(msg)
            time.sleep(interval)
        self.get_logger().info('[HOLD] 완료')

    def _make_joint_traj(self, target: list, start: list = None) -> JointTrajectory:
        if start is None:
            start = self._current_joints or list(HOME_JOINT_VALUES)
        max_delta = max(abs(t - s) for t, s in zip(target, start))
        duration  = max(3.0, max_delta / max(0.01, self._speed * 1.5))
        jt = JointTrajectory()
        jt.joint_names = list(HOME_JOINT_NAMES)
        pt0 = JointTrajectoryPoint()
        pt0.positions       = [float(v) for v in start]
        pt0.time_from_start = RosDuration(sec=0, nanosec=0)
        pt1 = JointTrajectoryPoint()
        pt1.positions       = [float(v) for v in target]
        sec  = int(duration)
        nsec = int((duration - sec) * 1e9)
        pt1.time_from_start = RosDuration(sec=sec, nanosec=nsec)
        jt.points = [pt0, pt1]
        self.get_logger().info(
            f'[TRAJ_DIRECT] max_Δ={max_delta:.3f} rad  dur={duration:.1f}s')
        return jt

    def _do_release_hand(self):
        start  = self._last_hand_enc if self._last_hand_enc else list(HAND_RELEASE_ENC)
        target = list(HAND_RELEASE_ENC)
        self.get_logger().info(f'[RELEASE] {HAND_STEPS}steps × {HAND_PERIOD}s')
        for i in range(1, HAND_STEPS + 1):
            t      = i / HAND_STEPS
            interp = [max(-32768, min(32767, int(round(s + t * (g - s)))))
                      for s, g in zip(start, target)]
            msg      = Int16MultiArray()
            msg.data = interp
            self._hand_pub.publish(msg)
            if i < HAND_STEPS:
                time.sleep(HAND_PERIOD)
        self._last_hand_enc = list(HAND_RELEASE_ENC)
        self.get_logger().info('[RELEASE] 완료')

    # ── Arm execution ─────────────────────────────────────────────────────────

    def _exec_arm(self, jt):
        self._exec_traj_smooth(jt)

    def _exec_traj_smooth(self, jt: JointTrajectory) -> list:
        if not jt.points:
            self.get_logger().error('[TRAJ] Empty trajectory')
            return []
        msg = JointTrajectory()
        msg.joint_names = list(jt.joint_names) if jt.joint_names else list(HOME_JOINT_NAMES)
        msg.points = jt.points
        spd = Float64()
        spd.data = max(0.001, min(1.0, float(self._speed)))
        self._franka_speed_pub.publish(spd)
        time.sleep(0.02)
        self._traj_smooth_pub.publish(msg)
        dur   = jt.points[-1].time_from_start.sec + \
                jt.points[-1].time_from_start.nanosec * 1e-9
        final = list(jt.points[-1].positions)
        self.get_logger().info(
            f'[TRAJ] sent {len(jt.points)} waypoints, duration={dur:.2f}s, '
            f'final={[round(v,4) for v in final]}')
        return final

    def _exec_traj(self, jt):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = jt
        future = self._traj_client.send_goal_async(goal)
        if not self._wait(future, 5.0):
            self.get_logger().error('Trajectory send timeout'); return
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('Trajectory rejected by controller'); return
        dur = 30.0
        if jt.points:
            p   = jt.points[-1]
            dur = p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
        rf = gh.get_result_async()
        if self._wait(rf, timeout=dur + 15.0):
            self.get_logger().info('Arm trajectory complete')
        else:
            self.get_logger().warning('Arm trajectory result timeout')

    def _exec_waypoints(self, jt):
        if not jt.points:
            self.get_logger().error('Empty trajectory'); return
        spd      = Float64()
        spd.data = max(0.001, min(1.0, float(self._speed)))
        self._franka_speed_pub.publish(spd)
        time.sleep(0.02)
        t_start = time.time()
        for i, pt in enumerate(jt.points):
            if len(pt.positions) != 7:
                self.get_logger().error(
                    f'Waypoint[{i}] positions len={len(pt.positions)}, expected 7')
                return
            t_target = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            t_now    = time.time() - t_start
            sleep_t  = t_target - t_now
            if sleep_t > 0:
                time.sleep(sleep_t)
            tgt      = Float64MultiArray()
            tgt.data = [float(v) for v in pt.positions]
            self._franka_target_pub.publish(tgt)
        self.get_logger().info(
            f'Waypoints sent: {len(jt.points)} pts, '
            f'final={[round(v,4) for v in jt.points[-1].positions]}')

    def _exec_direct(self, jt):
        if not jt.points:
            self.get_logger().error('Empty trajectory'); return
        final = jt.points[-1]
        if len(final.positions) != 7:
            self.get_logger().error(
                f'Expected 7 joint positions, got {len(final.positions)}'); return
        spd      = Float64()
        spd.data = max(0.001, min(1.0, float(self._speed)))
        self._franka_speed_pub.publish(spd)
        time.sleep(0.05)
        tgt      = Float64MultiArray()
        tgt.data = list(final.positions)
        self._franka_target_pub.publish(tgt)
        self.get_logger().info(f'Direct arm cmd sent: {[round(v,4) for v in tgt.data]}')

    # ── Hand execution ────────────────────────────────────────────────────────

    def _exec_hand(self, target_enc: list):
        start  = self._last_hand_enc if self._last_hand_enc else list(HAND_INIT_ENC)
        target = [int(v) for v in target_enc]
        total  = HAND_STEPS * HAND_PERIOD
        self.get_logger().info(
            f'[HAND] {HAND_STEPS}steps × {HAND_PERIOD}s = {total:.1f}s')
        self.get_logger().info(f'  target={target}')
        for i in range(1, HAND_STEPS + 1):
            t      = i / HAND_STEPS
            interp = [max(-32768, min(32767, int(round(s + t * (g - s)))))
                      for s, g in zip(start, target)]
            msg      = Int16MultiArray()
            msg.data = interp
            self._hand_pub.publish(msg)
            self.get_logger().info(f'  [{i:2d}/{HAND_STEPS}] t={t:.1f}  raw={interp}')
            if i < HAND_STEPS:
                time.sleep(HAND_PERIOD)
        self._last_hand_enc = target
        self.get_logger().info('[HAND] 파지 완료')

    # ── Main execution ────────────────────────────────────────────────────────

    def _run_guarded(self):
        try:
            self._execute()
        except Exception as e:
            self.get_logger().error(f'Unhandled exception: {e}')
            traceback.print_exc()
        finally:
            try:
                self._finalize()
            except Exception as e:
                self.get_logger().error(f'[FINALIZE] 예외 발생: {e}')
            rclpy.shutdown()

    def _execute(self):
        grasp  = self._summary['grasps'][0]
        bp     = grasp['base_pose']
        xyz_w  = bp['xyz']
        quat_w = bp['quat_xyzw']
        enc    = grasp['joint_angles_enc']
        T_wb   = self._summary.get('T_world_base')

        if T_wb is not None:
            xyz_b, quat_b = world_to_base(T_wb, xyz_w, quat_w)
            xyz_w_app     = [xyz_w[0], xyz_w[1], xyz_w[2] + self._approach_offset]
            xyz_b_app, quat_b_app = world_to_base(T_wb, xyz_w_app, quat_w)
        else:
            self.get_logger().warning('T_world_base not in summary — using xyz as-is in base frame')
            xyz_b, quat_b = xyz_w, quat_w
            xyz_b_app     = [xyz_w[0], xyz_w[1], xyz_w[2] + self._approach_offset]
            quat_b_app    = quat_w

        print('\n' + '=' * 60)
        print('  Grasp Executor')
        print(f'  EE target (world): xyz={[round(v,3) for v in xyz_w]}')
        print(f'  EE target (base) : xyz={[round(v,3) for v in xyz_b]}')
        print(f'  quat (base)      : {[round(v,3) for v in quat_b]}')
        print(f'  Hand enc (HW)    : {enc}')
        print(f'  Approach offset  : {self._approach_offset:.2f} m')
        print('\n  Grasp Executor  (STEP 1~4)')
        print(f'  Grasp (base): {[round(v,3) for v in xyz_b]}')
        print(f'  Approach z  : {xyz_b_app[2]:.3f} m')
        print('=' * 60)

        time.sleep(1.0)
        step_init_hand(self)
        step_go_home(self, confirm=False)
        target   = self._make_pose(*xyz_b,     *quat_b)
        approach = self._make_pose(*xyz_b_app, *quat_b_app)

        result = step_approach(self, approach, confirm=True)
        if result is None: return
        j1, self._approach_traj = result

        result = step_descend(self, target, seed=j1, confirm=True)
        if result is None: return
        j2, descend_traj = result

        print(f'\n  STEP 3/4  HAND 파지  enc={enc}')
        if not step_close_hand(self, enc, confirm=True): return

        j4 = step_lift(self, approach, seed=j2,
                       descend_traj=descend_traj, confirm=True)
        if j4 is None: return

        self._success = True
        print('\n  [OK] 파지 완료')

    def _finalize(self):
        print('\n' + '─' * 60)
        print('  [FINALIZE]')
        print('─' * 60)
        if self._success:
            print('  파지 완료 → 릴리즈 및 홈 복귀')
        else:
            print('  ⚠  실행이 중단되었습니다.')
        step_release_hand(self, confirm=True)
        step_go_home(self, confirm=True)
