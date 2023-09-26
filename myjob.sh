#!/bin/bash
#SBATCH --account=m22oc-S2343760
#SBATCH --job-name=mpi-nn
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2

# Load the appropriate modules, e.g.,
module load openmpi/4.1.4-cuda-11.8
module load nvidia/nvhpc-nompi/22.11

# Note the addition
export OMPI_MCA_pml=ob1
export LD_LIBRARY_PATH=/work/m22oc/m22oc/s2343760/mpi-nn/lib/armadillo/usr/lib64:/work/m22oc/m22oc/s2343760/mpi-nn/lib/lapack/lib64:$LD_LIBRARY_PATH


srun -n 2 ./main