#!/bin/bash
#SBATCH -n 12
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p zhuang,general,shared
#SBATCH -t 600
#SBATCH --mem=250000
#SBATCH -o F:/Harvard_BWH/projects/1001_MERFISH/MERmate/Examples/2021_08_02_U2OS_groupA/normalized_data_matlab/\log\barcode_metadata.out
#SBATCH -e F:/Harvard_BWH/projects/1001_MERFISH/MERmate/Examples/2021_08_02_U2OS_groupA/normalized_data_matlab/\log\barcode_metadata.err
#SBATCH --open-mode=truncate
#SBATCH --constraint=holyib

module load matlab/R2017a-fasrc02
mkdir -p /scratch/$USER/$SLURM_JOB_ID
matlab-default -nosplash -nodesktop -r "overwrite = true; nDPath = 'F:/Harvard_BWH/projects/1001_MERFISH/MERFISHpy/Examples/2021_08_02_U2OS_groupA/normalized_data_matlab/'; ExportBarcodeMetadata();"
rm -rf /scratch/$USER/$SLURM_JOB_ID
