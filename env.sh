#!/usr/bin/env bash
# BEVFormer 실습 환경 활성화 헬퍼
#   사용법:  source ~/bevformer_practice/env.sh
# ROS 오염 방지 + conda env + CUDA 11.1(conda) 경로를 한 번에 잡는다.

source "$HOME/miniconda3/etc/profile.d/conda.sh"
unset PYTHONPATH                      # §T12: ROS 등 PYTHONPATH 오염이 conda env를 덮는 것 방지
conda activate bevformer

export WORK="$HOME/bevformer_practice"
export BEV="$WORK/BEVFormer"
export NUSCENES="$BEV/data/nuscenes"
export CKPT="$BEV/ckpts/bevformer_tiny_epoch_24.pth"
export CONFIG="projects/configs/bevformer/bevformer_tiny.py"

# mmdet3d 등 CUDA 오퍼레이터 재빌드 시 필요 (§T2: conda gcc-9 + conda nvcc 11.1)
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export MPLBACKEND=Agg                 # 헤드리스 시각화 대비

echo "[bevformer env] python=$(which python)"
echo "[bevformer env] BEV=$BEV"
echo "[bevformer env] torch/cuda: $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>/dev/null)"
