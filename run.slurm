#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=0-10:00:00     # 10 minutes
#SBATCH --mail-user=cfk29@case.edu
#SBATCH -p gpu -C gpu2080 --gres=gpu:1
#SBATCH --job-name="Tumor bud"
#SBATCH --output="log.tb"

runId=001
lossType='F'
gpuId=1
pathPrefix=/scratch/users/cfk29/TB_detection/
layerNum=5
#13 dice, 14 tv, 15 ftv

module load cuda/10.0
module load gcc/6.3.0
module load python/3.6.6


#python3.6 -m pip install --user virtualenv
#python3.6 -m venv /scratch/users/cfk29/TB_detection/venv/
#source /scratch/users/cfk29/TB_detection/venv/bin/activate

#pip install future
#pip install tensorflow-gpu==1.13.1
#pip install keras==2.2.4
#pip install imageio
#pip install numpy
#pip install opencv-python==3.1.0.4

cd src
time python3.6 tumorBudMain.py $gpuId $runId tr $pathPrefix $layerNum $lossType &&
time python3.6 tumorBudMain.py $gpuId $runId ts $pathPrefix $layerNum $lossType ||
echo "[ERROR] Training step is failed! "

#deactivate
