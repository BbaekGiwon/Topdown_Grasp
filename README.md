# Grasp_fruit

KISTAR Franka FR3 + KISTAR Hand를 이용한 과일 파지/배치 시스템.  
RealSense 카메라로 RGB-D를 촬영하고, SAM3 세그멘테이션으로 대상 물체를 검출한 뒤,  
Top-down Grasp로 파지 위치를 계산하여 로봇이 자동으로 집어 내려놓는다.

---

## 시스템 구성

| 구성 요소 | 내용 |
|---|---|
| 로봇 | KISTAR Franka FR3 + KISTAR Hand |
| 카메라 | Intel RealSense D-series (RGB-D) |
| 소프트웨어 스택 | ROS2 Humble / MoveIt2 (Docker `ros2_humble`) |
| 파이프라인 환경 | Conda `pipeline_all` |
| 비전 모델 | SAM3 (text-prompted segmentation) |
| Grasp 알고리즘 | Top-down grasp (affordance_grasp) |

---

## 설치

### 1. Conda 환경 구성

```bash
cd HARILAB/Grasp_fruit
bash setup_pipeline_all.sh
conda activate pipeline_all
```

### 2. Docker 컨테이너 확인

ROS2 Humble + MoveIt2가 포함된 `ros2_humble` 컨테이너가 실행 중이어야 한다.

```bash
docker ps | grep ros2_humble
```

---

## 설정 파일

### `configs/paths.yaml` — 머신별 경로 설정

다른 PC에 클론할 경우 **이 파일만** 수정하면 된다.

```yaml
__CONDA_BASE__: /home/kist/miniforge3
__CONDA_ENV__:  pipeline_all
__DOCKER_CONTAINER__: ros2_humble
__ROS_DOMAIN_ID__:    9
__KISTAR_WS__:        /home/kist/HARILAB/dex_ros/isaac-ros/kistar_ws
__MOUNT_MAP__:
  - ["/home/kist/HARILAB", "/root/HARILAB"]
  - ["/home/kist/ros2_ws",  "/root/ros2_ws"]
```

### `configs/arm.yaml` — 로봇 팔 파라미터

| 파라미터 | 설명 |
|---|---|
| `home.joint_values` | HOME 자세 관절값 (rad) |
| `approach_offset_m` | 파지 위치 위에서 approach 시작 높이 (m) |
| `place_z_descent_m` | Place 시 HOME EE Z에서 하강 거리 (m) |
| `ee_correction.yaw_deg` | 핸드 장착 회전 오프셋 (°) |
| `ee_correction.x_offset_m` / `y_offset_m` | EE 위치 보정 (m) |

---

## 캘리브레이션

캘리브레이션 파일: `configs/calibration/extrinsic_20260612_170053.json`

| 필드 | 설명 | 변경 여부 |
|---|---|---|
| `T_base_camera` | 핸드-아이 캘리브레이션 결과 (카메라 → 베이스 변환) | **변경 금지** |
| `T_world_base` | 로봇 베이스 마운트 위치/자세 (world → base) | 자유롭게 변경 가능 |

### T_world_base 업데이트

로봇 베이스 위치가 바뀌었을 때 `update_world_base.py`로 행렬을 자동 계산하여 JSON에 기록한다.

```bash
# dry-run (미리보기)
python scripts/update_world_base.py \
    --calib configs/calibration/extrinsic_20260612_170053.json \
    --x 0.066 --y -0.122 --z 0.099 \
    --roll_deg 45.0 --dry_run

# 실제 적용
python scripts/update_world_base.py \
    --calib configs/calibration/extrinsic_20260612_170053.json \
    --x 0.066 --y -0.122 --z 0.099 \
    --roll_deg 45.0
```

---

## 사용법

### 인터랙티브 파이프라인 (권장)

SAM3 모델을 한 번 로드하고, 텍스트 쿼리를 반복 입력받아 실행한다.

```bash
conda activate pipeline_all
cd HARILAB/Grasp_fruit

# 비전만 (로봇 미실행)
python scripts/run_pipeline_interactive.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json

# Grasp 실행
python scripts/run_pipeline_interactive.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot

# Pick + Place 실행
python scripts/run_pipeline_interactive.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot --place

# MoveIt collision 비활성화 (로봇 베이스 이동 후 임시)
python scripts/run_pipeline_interactive.py \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot --place --disable_collision
```

`exit` / `quit` / `q` 입력 시 종료.

### 단발 파이프라인

```bash
# 카메라 캡처 + Grasp
python scripts/run_pipeline.py \
    --capture \
    --query "apple" \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot

# 카메라 캡처 + Pick + Place
python scripts/run_pipeline.py \
    --capture \
    --query "apple" \
    --calibration configs/calibration/extrinsic_20260612_170053.json \
    --execute_robot --place

# 저장된 NPZ 파일로 오프라인 처리
python scripts/run_pipeline.py \
    --input data/raw/scene_000.npz \
    --query "apple" \
    --calibration configs/calibration/extrinsic_20260612_170053.json
```

### 주요 공통 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--calibration` | 캘리브레이션 JSON 경로 | — |
| `--execute_robot` | 로봇 실행 활성화 | 비활성 |
| `--place` | Pick+Place 모드 (하강 거리는 arm.yaml 참조) | 비활성 |
| `--speed_factor` | 로봇 속도 배율 | 0.1 |
| `--approach_offset` | Approach 높이 오프셋 (m) | 0.10 |
| `--disable_collision` | MoveIt collision 검사 비활성화 (임시) | 비활성 |
| `--execute_mode` | 궤적 전송 방식 | `direct_franka_topic` |

---

## 파이프라인 단계

```
Stage 0: RealSense 캡처  →  data/raw/<stem>_000.npz
Stage 1: SAM3 추론       →  data/interim/<stem>_mask.png
Stage 2: Top-down Grasp  →  data/outputs/<stem>_topdown_summary.json
Stage 3: 로봇 실행       →  Docker exec → robot_executor.py
```

### Stage 3 로봇 실행 흐름

**Grasp 모드:**
1. HOME 이동 (확인 요청)
2. Approach 이동 (확인 요청)
3. 하강 (확인 요청)
4. 핸드 파지 (확인 요청)
5. 상승 (하강 역재생)
6. HOME 복귀 (approach 역재생)

**Place 모드 (Grasp 이후):**
1. HOME 이동 (확인 요청)
2. Approach → 하강 → 파지 → 상승 → HOME (상동)
3. HOME에서 Place 위치로 수직 하강 (Cartesian 계획)
4. 핸드 열기
5. 수직 상승 (하강 역재생)

---

## 디렉토리 구조

```
Grasp_fruit/
├── configs/
│   ├── arm.yaml                  # 로봇 팔 파라미터 (HOME 자세, 오프셋 등)
│   ├── hand.yaml                 # 핸드 파라미터
│   ├── paths.yaml                # 머신별 경로 설정
│   ├── calibration/              # 캘리브레이션 JSON 파일들
│   └── camera/
│       └── realsense.yaml        # RealSense 카메라 설정
│
├── scripts/
│   ├── run_pipeline_interactive.py  # 인터랙티브 파이프라인 (권장)
│   ├── run_pipeline.py              # 단발 파이프라인
│   ├── pipeline_core.py             # 공통 stage 함수 / argparse 빌더
│   ├── robot_executor.py            # Docker 내부 로봇 실행 진입점
│   ├── send_to_robot.py             # 호스트 → Docker exec 브릿지
│   ├── run_topdown_grasp.py         # Top-down grasp 계산
│   ├── run_sam3_only_stage.py       # SAM3 추론 (subprocess용)
│   ├── capture_realsense_once.py    # RealSense 단회 촬영
│   ├── update_world_base.py         # T_world_base 업데이트 유틸
│   ├── launch_moveit.py             # MoveIt 런치 헬퍼
│   ├── docker_runner.py             # Docker exec 유틸리티
│   └── utils/
│       ├── arm.py                   # arm.yaml 파싱, 상수 정의
│       ├── hand.py                  # hand.yaml 파싱
│       ├── paths.py                 # paths.yaml 파싱
│       ├── grasp.py                 # GraspExecutor (ROS2 Node)
│       ├── place.py                 # PlaceExecutor (GraspExecutor 상속)
│       └── step.py                  # 원자 step 함수들
│
├── src/
│   └── affordance_grasp/
│       ├── io/                      # NPZ 로드, 마스크 처리
│       └── geometry/                # 포인트클라우드, 변환 행렬 유틸
│
├── data/
│   ├── raw/                         # RealSense 캡처 NPZ
│   ├── interim/                     # SAM3 마스크, 오버레이
│   └── outputs/                     # Grasp summary JSON
│
├── docker/
│   ├── Dockerfile.moveit
│   └── entrypoint_moveit.sh
│
├── environment_pipeline_all.yml     # Conda 환경 정의
└── setup_pipeline_all.sh            # 환경 구성 스크립트
```

---

## 트러블슈팅

### IK 실패 (code=-31)

MoveIt collision scene이 로봇 베이스 위치와 맞지 않으면 IK가 차단된다.  
임시 해결: `--disable_collision` 플래그 사용.  
근본 해결: RViz Planning Scene에서 collision object 위치를 새 베이스 위치에 맞게 업데이트.

### Place가 HOME을 거치지 않고 바로 하강

`step_go_home` 완료 후 관절이 완전히 정착하기 전에 Cartesian 계획이 시작되면 발생할 수 있다.  
`scripts/utils/step.py`의 `step_place_from_home`에서 0.5초 settling wait + HOME_JOINT_VALUES seed로 처리됨.

### `--place_z_descent` 인식 불가 에러

이전 버전 옵션이다. `--place` (store_true)로 교체되었으며 하강 거리는 `configs/arm.yaml`의 `place_z_descent_m`으로 관리된다.
