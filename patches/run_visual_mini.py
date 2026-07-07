# mini + bevformer_tiny 용 시각화 러너 (원본 visual.py 함수 재사용)
# 원본 visual.py __main__ 은 v1.0-trainval + 특정 timestamp 경로가 하드코딩되어 있어(§T9)
# mini/우리 eval 산출물에 맞게 감싸서 호출한다.
#   실행: (cwd=$BEV) python tools/analysis_tools/run_visual_mini.py
import os, glob, sys
import mmcv
from nuscenes.nuscenes import NuScenes
import visual  # 같은 폴더의 원본 스크립트 (함수 정의만 로드, __main__ 은 실행 안 됨)

# 방금 Phase 3 평가가 남긴 nuScenes 제출 포맷 결과 (timestamp 자동 탐색)
cands = sorted(glob.glob('test/bevformer_tiny/*/pts_bbox/results_nusc.json'))
assert cands, "results_nusc.json 없음 — 먼저 Phase 3(dist_test.sh --eval bbox) 실행"
res_path = cands[-1]
print('[viz] results:', res_path)

# render_sample_data 등이 참조하는 모듈 전역 nusc 를 mini 로 세팅
visual.nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)

results = mmcv.load(res_path)
tokens = list(results['results'].keys())
n = min(int(sys.argv[1]) if len(sys.argv) > 1 else 6, len(tokens))

os.makedirs('viz', exist_ok=True)
for i in range(n):
    out = os.path.join('viz', f'sample_{i:02d}_{tokens[i][:8]}')
    visual.render_sample_data(tokens[i], pred_data=results, out_path=out)
    print(f'[viz] {i+1}/{n} -> {out}_camera.png')
print('[viz] done. 결과: $BEV/viz/*_camera.png (상단=예측, 하단=GT / 6개 카메라뷰)')
