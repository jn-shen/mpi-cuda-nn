#!/bin/bash
#SBATCH --account=m22oc-S2343760
#SBATCH --job-name=mpi-nn
#SBATCH --time=00:00:05
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

# Load the appropriate modules, e.g.,
module load openmpi/4.1.4-cuda-11.8
module load nvidia/nvhpc-nompi/22.2

export OMP_NUM_THREADS=1

# Note the addition
export OMPI_MCA_pml=ob1
export LD_LIBRARY_PATH=/work/m22oc/m22oc/s2343760/mpi-nn/lib/usr/lib64:$LD_LIBRARY_PATH

srun --ntasks=1 --cpus-per-task=2 --hint=nomultithread ./main -g 4