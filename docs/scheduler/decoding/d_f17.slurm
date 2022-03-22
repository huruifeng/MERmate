#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p shared,zhuang
#SBATCH -t 360
#SBATCH --mem=32000
#SBATCH -o F:/Harvard_BWH/projects/1001_MERFISH/MERmate/Examples/2021_08_02_U2OS_groupA/normalized_data_matlab/\log\decoding\fov_decode_17.out
#SBATCH -e F:/Harvard_BWH/projects/1001_MERFISH/MERmate/Examples/2021_08_02_U2OS_groupA/normalized_data_matlab/\log\decoding\fov_decode_17.err
#SBATCH --open-mode=append
#SBATCH --constraint=holyib
#SBATCH --exclude=holyzhuang01,holy2c14407

module load matlab/R2017a-fasrc02
matlab-default -nosplash -nodesktop -r "basePath = 'F:/Harvard_BWH/projects/1001_MERFISH/MERFISHpy/Examples/2021_08_02_U2OS_groupA/normalized_data_matlab/'; arrayID = 17; DecodeFOV; exit;"
