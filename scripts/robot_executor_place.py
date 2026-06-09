#!/usr/bin/env python3
"""
Pick → (Home → Place → Release → Home) executor.

실행 흐름:
  [확인 있음]  STEP 1-4: Approach → Target → Hand → Lift
  [자동 세트]  STEP 5:   HOME
               STEP 6:   PLACE APPROACH  (HOME EE XY, place_z + approach_offset)
               STEP 7:   PLACE DESCENT   (HOME EE XY, place_z)
               STEP 8:   RELEASE
               STEP 9:   HOME

place 위치 = HOME 에서 world Z 축으로 하강.
X,Y 는 HOME EE 위치 그대로, Z 만 --place_z 로 지정.

Usage:
    python3 scripts/robot_executor_place.py \\
        --summary_json data/outputs/scene_topdown_summary.json \\
        --place_z_descent 0.30 \\
        [--approach_offset 0.10]
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import rclpy
from rclpy.executors import MultiThreadedExecutor

from robot_executor import (
    GraspExecutor,
    world_to_base,
)
from utils.step import (
    step_approach, step_descend, step_lift,
    step_init_hand, step_close_hand, step_release_hand, step_go_home,
    step_place_from_home,
)


class PickPlaceViaHomeExecutor(GraspExecutor):
    """
    STEP 1-4 : Grasp (y/n 확인)
    STEP 5-9 : Home → Place → Release → Home (자동)
    """

    def __init__(self, summary, execute_mode, speed_factor,
                 approach_offset, place_z_descent, summary_json_path=''):
        self._place_z_descent = place_z_descent
        super().__init__(summary, execute_mode, speed_factor,
                         approach_offset, summary_json_path=summary_json_path)

    def _execute(self):
        grasp  = self._summary['grasps'][0]
        bp     = grasp['base_pose']
        xyz_w  = bp['xyz']
        quat_w = bp['quat_xyzw']
        enc    = grasp['joint_angles_enc']
        T_wb   = self._summary.get('T_world_base')

        if T_wb is not None:
            xyz_b, quat_b = world_to_base(T_wb, xyz_w, quat_w)
            xyz_b_app, quat_b_app = world_to_base(
                T_wb, [xyz_w[0], xyz_w[1], xyz_w[2] + self._approach_offset], quat_w)
        else:
            self.get_logger().warning('T_world_base 없음 — base frame 그대로 사용')
            xyz_b, quat_b = xyz_w, quat_w
            xyz_b_app, quat_b_app = (
                [xyz_w[0], xyz_w[1], xyz_w[2] + self._approach_offset], quat_w)

        print('\n' + '=' * 60)
        print('  Pick-Place-via-Home Executor  (STEP 1~9)')
        print(f'  Grasp (base): {[round(v,3) for v in xyz_b]}')
        print(f'  Place Z     : {self._place_z_descent:.3f} m  (HOME EE Z 기준 하강)')
        print('=' * 60)

        time.sleep(1.0)
        step_init_hand(self)              # 핸드 초기 자세 (j3=굽힘, 충돌 회피)
        step_go_home(self, confirm=False) # 팔 초기자세로 이동
        target   = self._make_pose(*xyz_b,     *quat_b)
        approach = self._make_pose(*xyz_b_app, *quat_b_app)

        # ── STEP 1-4: Grasp (y/n) ────────────────────────────────────────
        result = step_approach(self, approach, confirm=True)
        if result is None: return
        j1, approach_traj = result
        self._approach_traj = approach_traj   # _finalize 역재생용

        result = step_descend(self, target, seed=j1, confirm=True)
        if result is None: return
        j2, descend_traj = result

        print(f'\n  STEP 3/9  HAND 파지  enc={enc}')
        if not step_close_hand(self, enc, confirm=True): return

        j4 = step_lift(self, approach, seed=j2,
                       descend_traj=descend_traj, confirm=True)
        if j4 is None: return

        # ── STEP 5-9: 자동 세트 ───────────────────────────────────────────
        print('\n' + '─' * 60)
        print('  [AUTO] Home → Place → Release → Home')
        print('─' * 60)

        step_go_home(self, confirm=False, approach_traj=approach_traj)  # approach 역재생

        ok = step_place_from_home(self,
                                  place_z_descent=self._place_z_descent)
        if not ok:
            self.get_logger().error('Place 실패')
            return
        # step_place_from_home 내부에서 하강 역재생으로 HOME 복귀 완료
        step_init_hand(self)              # release 후 충돌 회피 자세로 복귀

        self._success = True
        print('\n  [OK] Pick-Place-via-Home 완료')

    def _finalize(self):
        print('\n' + '─' * 60)
        print('  [FINALIZE]')
        print('─' * 60)
        if self._success:
            print('  정상 완료.')
        else:
            print('  ⚠  실행이 중단되었습니다.')
            step_release_hand(self, confirm=True)
            step_go_home(self, confirm=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summary_json', required=True)
    p.add_argument('--execute_mode', default='direct_franka_topic',
                   choices=['trajectory_forwarder', 'direct_franka_topic'])
    p.add_argument('--speed_factor',    type=float, default=0.1)
    p.add_argument('--approach_offset', type=float, default=0.10)
    p.add_argument('--place_z_descent', type=float, required=True,
                   help='HOME EE Z 에서 내려갈 거리 (m). 양수 = 아래 방향.')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.summary_json) as f:
        summary = json.load(f)

    rclpy.init()
    node = PickPlaceViaHomeExecutor(
        summary,
        args.execute_mode,
        args.speed_factor,
        args.approach_offset,
        place_z_descent=args.place_z_descent,
        summary_json_path=args.summary_json,
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        node._hold_hand_position(duration=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(0 if node._success else 1)


if __name__ == '__main__':
    main()
