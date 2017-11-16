#!/bin/sh

#SBATCH --job-name=chieh
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH -p ser-par-10g-2
#SBATCH --output=./sbrunex.out
#SBATCH --error=./sbrunex.err

module load gnu-4.4-compilers
module load fftw-3.3.3
module load perl-5.20.0
module load slurm-14.11.8
module load platform-mpi
module load gnu-4.8.1-compilers
module load boost-1.55.0
module load oracle_java_1.7u40
module load perl-5.20.0
module load matlab_mdcs_2016a
module load hadoop-2.4.1
module load spark-1.4.1_hadoop_2.4
module load cuda-8.0

export PATH="/home/wu.chie/App/miniconda/bin:$PATH"

srun python autoencoder.py
