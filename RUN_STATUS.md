# BEVFormer 실습 프로젝트 — 실행 상태

출처 런북: `~/Downloads/BEVFormer_실행_Runbook.md`
머신: Ubuntu 22.04 / RTX 4060 Ti (8GB) / driver 580 (CUDA 13.0) — **컴파일은 conda CUDA 11.1 사용**
프로젝트 루트: `~/bevformer_practice`  ·  코드: `~/bevformer_practice/BEVFormer`

## 재개 방법
```bash
source ~/bevformer_practice/env.sh      # conda env + 경로 한 번에
cd $BEV
```

## 진행 로그
- [x] **Phase 0 환경**: GPU=RTX 4060 Ti / VRAM=8GB / driver=580(CUDA13). conda/git/gcc OK.
- [x] **Phase 1 스택 (★가장 어려움)**: torch **1.9.1**(cuda True) / mmcv **1.4.0** / mmdet **2.14.0** / mmseg **0.14.1** / mmdet3d **0.17.1** / numpy **1.19.5** / setuptools **59.5.0** — import 전부 OK.
  - 해결한 함정:
    - conda가 pytorch **CPU 빌드**를 잡음 → CUDA 빌드 문자열 `py3.8_cuda11.1_cudnn8.0.5_0`로 강제 재설치.
    - `cudatoolkit=11.1` 런타임엔 **nvcc 없음** → `cudatoolkit-dev=11.1.1`(conda-forge)로 nvcc 11.1 추가(§T2).
    - 시스템 gcc 11 ↔ CUDA11.1 비호환 → conda `gxx/gcc_linux-64=9`로 host 컴파일러 지정(§T2).
    - defaults 채널 ToS → `conda tos accept`(§T10).
  - mmdet3d 0.17.1 CUDA 오퍼레이터 소스 빌드 성공(에러 0).
- [x] **Phase 2a 데이터 배치**: `v1.0-mini`·`samples`·`sweeps`는 기존 추출본(`~/nuscenes/mini`) **심링크**, `maps/`는 복사 후 **map-expansion v1.3** 병합.
- [x] **Phase 2c 가중치**: `bevformer_tiny_epoch_24.pth` (383MB) → `BEVFormer/ckpts/`.
- [x] **Phase 2a can_bus**: `can_bus.zip`(745M) → `data/can_bus/`.
- [x] **Phase 2b info(.pkl)**: `create_data.py ... --version v1.0-mini --canbus ./data` → `nuscenes_infos_temporal_{train,val}.pkl` 생성.
- [x] **Phase 3 평가**: **NDS=0.3253 / mAP=0.2649** (mini val, tiny). car 0.478·bus 0.575·ped 0.447.
- [x] **Phase 4 시각화**: `BEVFormer/viz/*_camera.png`(6뷰 카메라, 파랑=예측/초록=GT) + `*_bev.png`(BEV) 6샘플.
- [x] **Phase 5 짧은 학습**: loss **20.66 → 14.61** 우하향 (3 epoch, VRAM 3530MB). `work_dirs/bevformer_tiny_mini/`.

## ⚠️ 런북(draft)이 빠뜨린 항목 — 이 머신에서 추가로 해결한 것
런북은 status:draft(미검증)라 아래는 문서에 없던 실제 함정. 다음 실행 시 시간 단축용:
1. **pytorch CPU 빌드 오선택** — `conda install pytorch==1.9.1 ...` 가 conda-forge CPU 빌드를 잡음. → 빌드 문자열 `=py3.8_cuda11.1_cudnn8.0.5_0` 명시.
2. **nvcc 부재** — `cudatoolkit=11.1`(런타임)엔 nvcc 없음. → `cudatoolkit-dev=11.1.1`(conda-forge) 추가.
3. **detectron2 누락** — 플러그인 dd3d가 import. → prebuilt `detectron2 -f .../cu111/torch1.9/index.html`.
4. **Pillow 10 비호환** — detectron2가 `Image.LINEAR` 참조(제거됨). → `Pillow==9.5.0`.
5. **yapf 0.43 비호환** — mmcv 1.4.0 config dump가 `FormatCode(verify=)` 사용(제거됨). → `yapf==0.32.0` (train.py 필수).
6. **test.py 단일 GPU 경로 봉인** — `assert False`. `--out`도 봉인. → 평가는 `dist_test.sh <CONFIG> <CKPT> 1 --eval bbox`(1-GPU distributed).
7. **`projects` 모듈 인식** — 직접 실행 시 `export PYTHONPATH=$BEV` 필요(dist_test.sh는 자동). env.sh는 안전을 위해 unset만; 실행 시 `export PYTHONPATH=$BEV`.
8. **visual.py 하드코딩(§T9)** — v1.0-trainval + 특정 timestamp 경로 하드코딩, CLI 인자 없음. → `tools/analysis_tools/run_visual_mini.py` 래퍼로 대체(mini + 최신 results_nusc.json 자동 탐색).

## 재실행 요약 (모든 Phase 통과 기준)
```bash
source ~/bevformer_practice/env.sh && cd $BEV && export PYTHONPATH=$BEV
# 평가:  bash tools/dist_test.sh $CONFIG $CKPT 1 --eval bbox
# 시각화: python tools/analysis_tools/run_visual_mini.py 6   # → viz/*.png
# 학습:  python tools/train.py $CONFIG --work-dir work_dirs/bevformer_tiny_mini
```
