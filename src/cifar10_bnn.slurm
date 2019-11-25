#!/bin/bash
#SBATCH --nodelist=node5-7
#SBATCH --ntasks=1
#SBATCH --partition=GTX780
#SBATCH --mail-user=vladimir.shadymov@gmail.com
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --job-name=c10_bnn

module load cuda/10.0
srun python3 -u cifar10_conv_bnn.py --batch-size=128 --lr=1e-3 --epoch=500 --save-model --cuda-num=0 > cifar10_bnn.log
