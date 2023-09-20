module load openmpi/4.1.4-cuda-11.8
module load nvidia/nvhpc-nompi/22.11

# export LD_LIBRARY_PATH=/work/m22oc/m22oc/s2343760/mpi-nn/lib/usr/lib64:$LD_LIBRARY_PATH
mpirun -n 36 ./main -g