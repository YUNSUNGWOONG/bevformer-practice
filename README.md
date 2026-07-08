# bevformer-practice

BEVFormer(**tiny**)를 nuScenes **mini**로 실제 실행/검증한 실습 프로젝트 하네스.
설치 → 정량 평가(NDS/mAP) → 시각화(카메라+BEV) → 짧은 학습 → temporal ablation 까지 한 바퀴 돈다.

> 검증 환경: Ubuntu 22.04 / RTX 4060 Ti (8GB) / driver 580. CUDA 11.1 컴파일은 전부 conda로 처리(시스템 드라이버 불변).

## 결과 (2026-07-07 검증)
- 스택: `torch 1.9.1(cuda) · mmcv 1.4.0 · mmdet 2.14.0 · mmseg 0.14.1 · mmdet3d 0.17.1 · numpy 1.19.5`
- **Phase 3 평가**: NDS **0.3253** / mAP **0.2649** (mini val, tiny)
- **Phase 5 학습**: loss **20.66 → 14.61** (우하향), VRAM 3530MB
- **Ablation**: `video_test_mode=False`(temporal OFF) → NDS **−29%**, mAVE **+42%** — 시간 융합/CAN bus의 값어치

## 레포에 든 것 (하네스만 — 대용량은 .gitignore)
| 파일 | 내용 |
|---|---|
| `env.sh` | conda env + 경로(CUDA 11.1) 한 번에 진입 |
| `RUN_STATUS.md` | Phase 0~5 결과 + 공식 런북이 빠뜨려 해결한 8가지 함정(§T13~T18) |
| `LAB_GUIDE.md` | 기존 환경으로 Phase 3~5만 직접 돌리는 강사용 실습 가이드 |
| `docs/BEVFormer_실행_Runbook.md` | status draft → **validated** 로 갱신한 런북 스냅샷 |
| `patches/run_visual_mini.py` | mini용 시각화 래퍼(원본 `visual.py` 하드코딩 우회) |

> 업스트림 클론(`BEVFormer/`, `mmdetection3d/`)·nuScenes 데이터·가중치·출력물은 `.gitignore`로 제외. 재현은 `docs/BEVFormer_실행_Runbook.md` 대로.

## 시작
```bash
source ~/bevformer_practice/env.sh && cd $BEV && export PYTHONPATH=$BEV
# 평가:  bash tools/dist_test.sh $CONFIG $CKPT 1 --eval bbox
```

## 관련 레포

- 🧭 [**nuscenes-canbus-lab**](https://github.com/YUNSUNGWOONG/nuscenes-canbus-lab) — 자매 레포. 이 프로젝트의 ablation에서 드러난 **temporal 정렬** 개념만 떼어, GPU·모델 없이 CAN bus ego-pose 시계열로 손으로 재현하는 가벼운 실습(궤적 · SE(2) 정렬 · 속도 복원 · 센서 교차검증) + 개념 정리 PDF.
