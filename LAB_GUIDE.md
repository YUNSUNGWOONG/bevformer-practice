# BEVFormer 실습 가이드 (강사 사전 리허설용)

> **대상**: 이미 설치·검증이 끝난 `bevformer` conda 환경.  **설치(Phase 0~2)는 다시 하지 않는다.**
> **목표**: Phase 3(평가) → 4(시각화) → 5(짧은 학습)을 직접 손으로 돌려 "돌아간다 + BEV가 나온다"를 체험.
> **소요**: 대략 평가 2분 + 시각화 2분 + 학습 5~7분 = **~15분**.  GPU: RTX 4060 Ti 8GB로 충분(피크 ~3.5GB).
> 상세 원리·설치는 `~/Downloads/BEVFormer_실행_Runbook.md`, 함정은 그 문서 §T13~T18 참고.

---

## 0. 매 터미널 세션 시작 시 (필수)

새 터미널을 열 때마다 **한 번씩** 실행. 이 프로젝트의 모든 명령은 이걸 먼저 한 상태를 가정한다.

```bash
source ~/bevformer_practice/env.sh   # conda activate bevformer + 경로(CUDA 11.1) 세팅
cd $BEV                              # = ~/bevformer_practice/BEVFormer
export PYTHONPATH=$BEV               # 'projects' 플러그인 인식용 (env.sh는 안전상 PYTHONPATH를 비워둠)
```

**✅ 여기서 나와야 정상** (env.sh가 자동 출력):
```
[bevformer env] python=/home/admin2/miniconda3/envs/bevformer/bin/python
[bevformer env] torch/cuda: 1.9.1 True        ← True 여야 함. False면 §T13 (설치 문제)
```

> 💡 **강사 팁**: 실습생이 가장 많이 틀리는 지점 1위가 "터미널 새로 열고 `source` 안 함" → `conda: command not found`나 `No module named ...`. 매번 0번을 강조할 것.

---

## 1. Phase 3 — 정량 평가 (NDS / mAP) · "돌아간다" 1차 확인

사전학습 tiny 가중치로 nuScenes **mini val**을 평가한다.

```bash
export PORT=29511
bash tools/dist_test.sh $CONFIG $CKPT 1 --eval bbox
```

- ⚠️ `python tools/test.py ...` 로 직접 돌리지 말 것 — 이 리비전은 단일 GPU 경로가 막혀 `AssertionError`가 난다(§T18). **반드시 `dist_test.sh ... 1`**(마지막 1 = GPU 1장).
- 2~3분 소요. 경고(UserWarning/deprecated)는 무시.

**✅ 이런 표가 나오면 정상** (숫자는 검증 실행 기준, 소수점 근처면 OK):
```
mAP: 0.2649
mATE: 0.8562 ... 
NDS: 0.3253
Per-class:  car 0.478 · bus 0.575 · pedestrian 0.447 · truck 0.390 ...
```

> 💡 **강사 관찰 포인트**
> - 가중치는 **full trainval**로 학습됐는데 여기선 **mini val**로 평가한다. 그래서 논문값(tiny ~NDS 0.35)과 **정확히 같지 않은 게 정상**. 목적은 정확도가 아니라 "파이프라인이 끝까지 도는가".
> - `trailer`·`construction_vehicle`가 AP 0.000인 건 mini에 해당 객체가 거의 없어서다 → 실습생 예상 질문.
> - 결과 json은 `test/bevformer_tiny/<timestamp>/pts_bbox/results_nusc.json`에 저장된다(다음 단계에서 재사용).

---

## 2. Phase 4 — 시각화 (카메라 6뷰 + BEV 예측)

Phase 3가 남긴 결과 json을 그대로 재사용해 이미지를 만든다. **새로 추론하지 않는다.**

```bash
export MPLBACKEND=Agg    # 헤드리스(파일 저장) 모드
python tools/analysis_tools/run_visual_mini.py 6
```

- ⚠️ 원본 `visual.py`는 경로가 하드코딩(v1.0-trainval + 특정 timestamp)돼 있어 그대로는 안 된다. 위 `run_visual_mini.py`는 mini + 최신 `results_nusc.json`을 자동으로 잡는 래퍼다(§T18).
- 인자 `6` = 렌더할 샘플 수.

**✅ 결과 확인**:
```bash
ls -lh viz/          # sample_00_*_camera.png (6뷰) + sample_00_*_bev.png (BEV) ...
```
- `*_camera.png`: 6개 카메라. **상단=예측(파랑 박스), 하단=GT(초록 박스).**
- `*_bev.png`: LiDAR 포인트(높이별 색) 위에 예측(파랑)/GT(초록) 3D 박스 조감도.
- 이미지 뷰어로 열기: `xdg-open viz/sample_00_*_bev.png` (GUI 있을 때)

> 💡 **강사 관찰 포인트**: 파랑(예측)이 초록(GT)과 얼마나 겹치는지 보게 하라. tiny+mini라 완벽하진 않지만 차량 위치는 대체로 맞는다 → "카메라만으로 BEV 박스가 나온다"는 BEVFormer의 핵심 체감 포인트.

---

## 3. Phase 5 — 짧은 학습 (loss가 내려가는지)

정확도가 목표가 아니라 **학습 루프가 돌고 loss가 우하향**하는지 확인. 몇 분 돌리고 중단한다.

```bash
timeout --signal=INT 420 python tools/train.py $CONFIG \
  --work-dir work_dirs/bevformer_tiny_mini
```

- `timeout ... 420` = 7분 후 자동으로 Ctrl+C(정상 중단). 직접 볼 거면 그냥 `python tools/train.py ...` 후 원할 때 Ctrl+C.
- ⚠️ 시작하자마자 `TypeError: FormatCode() ... 'verify'`로 죽으면 yapf 문제(§T17) — 이 환경은 이미 `yapf==0.32.0`으로 맞춰져 있어 정상 동작.

**✅ 이렇게 loss가 내려가면 정상**:
```
Epoch [1][50/323]  ... loss: 20.66
Epoch [1][300/323] ... loss: 17.47
Epoch [2][300/323] ... loss: 15.38
Epoch [3][150/323] ... loss: 14.61      ← 우하향 확인
memory: 3530 (MB)  ← 8GB 안에서 넉넉
```
로그 파일: `work_dirs/bevformer_tiny_mini/<날짜>.log` (`grep "loss:" <로그>`로 추세만 봐도 됨).

> 💡 **강사 관찰 포인트**
> - mini는 데이터가 적어 loss가 들쭉날쭉·과적합하는 게 정상. "완만하게라도 내려가면 성공".
> - 원 config는 24 epoch 가정. 실습에선 **수십~수백 iter만 보고 중단**해도 목적 달성.
> - OOM 나면(다른 GPU 프로세스 점유 시) config의 `samples_per_gpu`를 1(이미 1), `queue_length`를 줄인다(§T8).

---

## 4. 실습 후 정리 (선택)

디스크/GPU 정리가 필요하면:
```bash
rm -rf work_dirs/bevformer_tiny_mini    # 학습 로그·체크포인트
rm -rf viz test                         # 시각화·평가 산출물
nvidia-smi                              # 남은 GPU 프로세스 확인
```
> 데이터(`data/`)·가중치(`ckpts/`)·conda 환경은 **지우지 말 것** — 실습생용으로 그대로 재사용한다.

---

## 한눈에 보기 (치트시트)

| 단계 | 명령 | 정상 신호 |
|---|---|---|
| 0. 세션 | `source ~/bevformer_practice/env.sh && cd $BEV && export PYTHONPATH=$BEV` | `torch/cuda: 1.9.1 True` |
| 3. 평가 | `bash tools/dist_test.sh $CONFIG $CKPT 1 --eval bbox` | `NDS: ~0.325 / mAP: ~0.265` |
| 4. 시각화 | `python tools/analysis_tools/run_visual_mini.py 6` | `viz/*_camera.png`, `*_bev.png` 생성 |
| 5. 학습 | `timeout --signal=INT 420 python tools/train.py $CONFIG --work-dir work_dirs/bevformer_tiny_mini` | `loss: 20→14` 우하향 |

문제 생기면 → 런북 §트러블슈팅 T1~T18. 이 환경 특유의 함정은 **T13~T18**.
