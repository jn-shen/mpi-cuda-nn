# MPI-CUDA-NN

## Project Overview
This project, originating from Stanford University's CME-213 course, aims to implement a simple two-layer neural network using CUDA and to parallelize the training process with MPI for the recognition of handwritten digit images. The project employs a data parallel approach, parallelizing each iteration that processes a batch of images and performing a reduction on the gradients at the end of each iteration.

To ensure the correctness of the implementation, the output of the project has successfully passed cross-validation with a CPU serial program, demonstrating good accuracy and reliability. This lays a solid foundation for further optimization and expansion.

## Dependencies

- CUDA (NVHPC 22.11)
- OpenMPI
- Armadillo

## Code Structure
- main.cpp: Main file for the project.
- gpu_func.cu, inc/gpu_func.h: Implementation and declaration of GPU CUDA wrapper functions and kernels.
- neural_network.cpp, inc/neural_network.h: Implementation of the neural network.
- utils/: Contains utility files for tests, common operations, and data reading.
- obj/: Stores object files generated during compilation.

## Run the program
```
mpirun -np <number_of_processes> ./myGEMM <input_arguments>
```
We provide several useful command line arguments for main:
- -n int to change number of neurons in the hidden layer to num
- -r float, -l float to change reg and learning_rate
- -e int, -b int for num_epochs and batch_size
- -s to run the sequential training together with your parallel training to compare their performance.
- -d for the debug mode.
- -p int to print debug output and files every num iterations.
- -g int for grading mode. Options are 1, 2, 3, 4. Options 1, 2, 3 run the three test cases for
checking correctness, and option 4 runs the GEMM case.
