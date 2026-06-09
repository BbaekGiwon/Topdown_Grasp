#!/usr/bin/env python3
"""
send_to_ros2.py — DRO-Grasp 출력을 ROS2 공유 디렉터리에 씁니다.

사용법:
    python scripts/send_to_ros2.py \\
        --grasp_json  data/outputs/<stem>_summary.json \\
        --shared_dir  /home/kist/shared/grasp_ros2 \\
        [--grasp_index 0]

    # run_pipeline.py 이후 자동으로 실행하는 예시:
    python scripts/run_pipeline.py --input scene.npz --instruction "..." && \\
    python scripts/send_to_ros2.py --grasp_json data/outputs/<stem>_summary.json

출력 (shared_dir/grasp_command.json):
    {
      "status"      : "ready",
      "robot"       : "kistar",
      "grasp_index" : 0,
      "source_json" : "<stem>_summary.json",
      "pose": {                          # 로봇 base frame (calibration 있을 때)
          "frame"      : "base",
          "xyz"        : [x, y, z],
          "quat_xyzw"  : [qx, qy, qz, qw]
      },
      "hand_enc" : [int16 × 16]          # ROS2 HW 순서: [thumb, index, middle, ring]
    }

Joint 순서 변환:
    DRO 내부 순서 : [index×4, middle×4, ring×4, thumb×4]
    ROS2 HW 순서  : [thumb×4, index×4, middle×4, ring×4]
    rad → enc     : round(rad × 8192 / π)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# DRO 내부 JSON 순서 → ROS2 HW 순서 인덱스 매핑
# DRO: [index(0-3), middle(4-7), ring(8-11), thumb(12-15)]
# ROS2: [thumb(0-3), index(4-7), middle(8-11), ring(12-15)]
_JSON_TO_ROS2 = [12, 13, 14, 15,   # thumb
                  0,  1,  2,  3,   # index
                  4,  5,  6,  7,   # middle
                  8,  9, 10, 11]   # ring

RAD_TO_ENC = 8192.0 / math.pi


def rad_to_enc(rad: float) -> int:
    return int(round(rad * RAD_TO_ENC))


def pick_best_grasp(summary: dict, index: int) -> dict:
    grasps = summary.get("grasps", [])
    if not grasps:
        raise ValueError("summary JSON에 'grasps' 항목이 없습니다.")
    if index >= len(grasps):
        raise ValueError(f"grasp_index={index}가 범위를 초과합니다 (총 {len(grasps)}개).")
    return grasps[index]


def extract_pose(grasp: dict, summary: dict) -> dict:
    """base_pose 우선 사용, 없으면 camera-frame pose 반환."""
    if "base_pose" in grasp:
        bp = grasp["base_pose"]
        return {
            "frame":     "base",
            "xyz":       bp["xyz"],
            "quat_xyzw": bp["quat_xyzw"],
        }

    # base_pose 없음 → camera frame fallback
    print("[WARN] 'base_pose' 없음: calibration JSON이 설정되지 않았습니다.")
    print("       DRO 실행 시 --calibration 옵션을 추가하세요.")
    print("       camera frame pose로 진행합니다 (좌표 오류 주의).")
    xyz = grasp.get("translation", [0.0, 0.0, 0.0])
    euler = grasp.get("euler_xyz", [0.0, 0.0, 0.0])
    # euler_xyz → quaternion
    rx, ry, rz = euler
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return {
        "frame":     "camera",
        "xyz":       list(xyz),
        "quat_xyzw": [qx, qy, qz, qw],
    }


def extract_hand_enc(grasp: dict) -> list[int]:
    """joint_angles (DRO 순서, rad) → ROS2 HW 순서 int16 encoder."""
    angles = grasp.get("joint_angles")
    if angles is None:
        raise ValueError("grasp에 'joint_angles'가 없습니다.")
    if len(angles) != 16:
        raise ValueError(f"joint_angles 길이가 {len(angles)}입니다 (16 필요).")
    ros2_angles = [angles[i] for i in _JSON_TO_ROS2]
    return [rad_to_enc(a) for a in ros2_angles]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grasp_json",   required=True,
                   help="DRO-Grasp summary JSON 경로 (*_summary.json)")
    p.add_argument("--shared_dir",   default="/home/kist/shared/grasp_ros2",
                   help="호스트↔Docker 공유 디렉터리 (기본: /home/kist/shared/grasp_ros2)")
    p.add_argument("--grasp_index",  type=int, default=0,
                   help="사용할 grasp 인덱스 (기본: 0 = 최고 점수)")
    args = p.parse_args()

    # 입력 JSON 읽기
    grasp_json_path = Path(args.grasp_json)
    if not grasp_json_path.exists():
        print(f"[ERROR] 파일 없음: {grasp_json_path}")
        sys.exit(1)

    with open(grasp_json_path) as f:
        summary = json.load(f)

    robot = summary.get("robot_name", "unknown")
    print(f"Robot: {robot}  |  총 grasp: {len(summary.get('grasps', []))}개")

    # Grasp 선택
    grasp = pick_best_grasp(summary, args.grasp_index)
    print(f"Grasp #{args.grasp_index}  score={grasp.get('score', 'N/A')}")

    # Pose 추출
    pose = extract_pose(grasp, summary)
    xyz = pose["xyz"]
    quat = pose["quat_xyzw"]
    print(f"Pose [{pose['frame']}]:  xyz=({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")
    print(f"               quat=({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")

    # Hand encoder 변환
    hand_enc = extract_hand_enc(grasp)
    print(f"Hand enc (ROS2 순서): {hand_enc}")

    # 공유 디렉터리 생성 및 grasp_command.json 쓰기
    shared_dir = Path(args.shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)

    command = {
        "status":      "ready",
        "robot":       robot,
        "grasp_index": args.grasp_index,
        "source_json": str(grasp_json_path),
        "pose":        pose,
        "hand_enc":    hand_enc,
    }

    out_path = shared_dir / "grasp_command.json"
    tmp_path = shared_dir / "grasp_command.json.tmp"

    with open(tmp_path, "w") as f:
        json.dump(command, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)  # atomic rename

    print(f"\n[OK] {out_path} 에 기록 완료.")
    print("     Docker ROS2 노드(grasp_executor)가 파일을 감지하면 실행됩니다.")


if __name__ == "__main__":
    main()
