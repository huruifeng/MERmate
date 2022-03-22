#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p zhuang,general,shared
#SBATCH -t 720
#SBATCH --mem=127000
#SBATCH -o F:/Harvard_BWH/projects/1001_MERFISH/MERmate/Examples/2021_08_02_U2OS_groupA/normalized_data_matlab/\log\mosaic.out
#SBATCH -e F:/Harvard_BWH/projects/1001_MERFISH/MERmate/Examples/2021_08_02_U2OS_groupA/normalized_data_matlab/\log\mosaic.err
#SBATCH --open-mode=append
#SBATCH --constraint=holyib

module load matlab/R2017a-fasrc02
matlab-default -nosplash -nodesktop -r "nDPath = 'F:/Harvard_BWH/projects/1001_MERFISH/MERFISHpy/Examples/2021_08_02_U2OS_groupA/normalized_data_matlab/'; LowResMosaic();"
