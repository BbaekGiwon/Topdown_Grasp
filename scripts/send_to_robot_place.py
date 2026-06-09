#!/usr/bin/env python3
"""
Send grasp + place to robot via Docker container.
Pick → Home → Place (HOME Z 하강) → Release → Home.

Usage:
    python scripts/send_to_robot_place.py \\
        --summary_json data/outputs/scene_topdown_summary.json \\
        --place_z_descent 0.15
"""

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from utils.paths import KISTAR_WS as DEFAULT_KISTAR_WS

from docker_runner import (
    DOCKER_CONTAINER,
    to_container_path, ensure_running,
    ros_exec_cmd, run_in_container,
    ask_and_record, stop_recording,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary_json",  required=True)
    p.add_argument("--execute_mode",  default="direct_franka_topic",
                   choices=["trajectory_forwarder", "direct_franka_topic"])
    p.add_argument("--speed_factor",    type=float, default=0.1)
    p.add_argument("--approach_offset", type=float, default=0.10)
    p.add_argument("--place_z_descent", type=float, required=True,
                   help="HOME EE Z 에서 내려갈 거리 (m). 양수 = 아래 방향.")
    p.add_argument("--container",  default=DOCKER_CONTAINER)
    p.add_argument("--kistar_ws",  default=DEFAULT_KISTAR_WS)
    return p.parse_args()


def main():
    args = parse_args()

    summary_host  = str(Path(args.summary_json).resolve())
    executor_ctr  = to_container_path(str(SCRIPTS / "robot_executor_place.py"))
    summary_ctr   = to_container_path(summary_host)
    kistar_ws_ctr = to_container_path(args.kistar_ws)

    print(f"[send_to_robot_place]")
    print(f"  summary      : {summary_host}")
    print(f"  place_z_descent: {args.place_z_descent} m (HOME EE Z 기준 하강)")

    if not Path(summary_host).exists():
        print(f"[ERROR] summary_json 없음: {summary_host}")
        sys.exit(1)

    ensure_running(args.container)
    stop_event, thread, _ = ask_and_record()

    bash_cmd = ros_exec_cmd(
        executor_ctr, summary_ctr, kistar_ws_ctr,
        extra_args=(
            f"--execute_mode {args.execute_mode} "
            f"--speed_factor {args.speed_factor} "
            f"--approach_offset {args.approach_offset} "
            f"--place_z_descent {args.place_z_descent}"
        ),
    )
    rc = run_in_container(args.container, bash_cmd)
    stop_recording(stop_event, thread)

    if rc == 0:
        print("\n[send_to_robot_place] 완료.")
    else:
        print(f"\n[send_to_robot_place] 종료 코드: {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
