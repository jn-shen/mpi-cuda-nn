#include "neural_network.h"

#include <cuda_runtime.h>
#include <mpi.h>
#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "iomanip"

#include "utils/common.h"

real norms(NeuralNetwork &nn)
{
  real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i)
  {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork &nn, int iter)
{
  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  nn.W[0].save(s.str(), arma::raw_ascii);
  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  nn.W[1].save(t.str(), arma::raw_ascii);
  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  nn.b[0].save(u.str(), arma::raw_ascii);
  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork &nn, int iter,
                        std::ofstream &error_file)
{
  arma::Mat<real> A, B, C, D;

  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  A.load(s.str(), arma::raw_ascii);
  real max_errW0 = arma::norm(nn.W[0] - A, "inf") / arma::norm(A, "inf");
  real L2_errW0 = arma::norm(nn.W[0] - A, 2) / arma::norm(A, 2);

  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  B.load(t.str(), arma::raw_ascii);
  real max_errW1 = arma::norm(nn.W[1] - B, "inf") / arma::norm(B, "inf");
  real L2_errW1 = arma::norm(nn.W[1] - B, 2) / arma::norm(B, 2);

  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  C.load(u.str(), arma::raw_ascii);
  real max_errb0 = arma::norm(nn.b[0] - C, "inf") / arma::norm(C, "inf");
  real L2_errb0 = arma::norm(nn.b[0] - C, 2) / arma::norm(C, 2);

  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  D.load(v.str(), arma::raw_ascii);
  real max_errb1 = arma::norm(nn.b[1] - D, "inf") / arma::norm(D, "inf");
  real L2_errb1 = arma::norm(nn.b[1] - D, 2) / arma::norm(D, 2);

  int ow = 15;

  if (iter == 0)
  {
    error_file << std::left << std::setw(ow) << "Iteration" << std::left
               << std::setw(ow) << "Max Err W0" << std::left << std::setw(ow)
               << "Max Err W1" << std::left << std::setw(ow) << "Max Err b0"
               << std::left << std::setw(ow) << "Max Err b1" << std::left
               << std::setw(ow) << "L2 Err W0" << std::left << std::setw(ow)
               << "L2 Err W1" << std::left << std::setw(ow) << "L2 Err b0"
               << std::left << std::setw(ow) << "L2 Err b1"
               << "\n";
  }

  error_file << std::left << std::setw(ow) << iter << std::left << std::setw(ow)
             << max_errW0 << std::left << std::setw(ow) << max_errW1
             << std::left << std::setw(ow) << max_errb0 << std::left
             << std::setw(ow) << max_errb1 << std::left << std::setw(ow)
             << L2_errW0 << std::left << std::setw(ow) << L2_errW1 << std::left
             << std::setw(ow) << L2_errb0 << std::left << std::setw(ow)
             << L2_errb1 << "\n";
}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork &nn, const arma::Mat<real> &X,
                 struct cache &cache)
{
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";tw
  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

/* GPU IMPLEMENTATIONS */
void GPUfeedforward(GPU_NN nn, GPU_cache bpcache, real *d_X_batch, int n_clos)
{
  real alpha = 1.0;
  real beta = 1.0;

  GPUrepmat(nn.d_b0, bpcache.d_z1, nn.n_rows_0, n_clos);

  myGEMM(nn.d_W0, d_X_batch, bpcache.d_z1, &alpha, &beta, nn.n_rows_0,
         n_clos, nn.n_cols_0);

  GPUsigmoid(bpcache.d_z1, bpcache.d_a1, nn.n_rows_0, n_clos);

  GPUrepmat(nn.d_b1, bpcache.d_z2, nn.n_rows_1, n_clos);

  myGEMM(nn.d_W1, bpcache.d_a1, bpcache.d_z2, &alpha, &beta, nn.n_rows_1,
         n_clos, nn.n_rows_0);

  GPUsoftmax(bpcache.d_z2, bpcache.d_yc, nn.n_rows_1, n_clos);
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork &nn, const arma::Mat<real> &y, real reg,
              const struct cache &bpcache, struct grads &bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<real> da1 = nn.W[1].t() * diff;

  arma::Mat<real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

void GPUbackprop(GPU_NN nn, real reg, GPU_cache bpcache, GPU_grads bpgrads, BP_temp temp,
                 Matrix_t mat_t, real *d_x_batch, real *d_y_batch, int n_clos, int batch_clos)
{
  real alpha = 1.0;
  real beta = 0.0;

  GPUadd(bpcache.d_yc, d_y_batch, temp.d_diff, 1.0 / batch_clos, -1.0 / batch_clos,
         nn.n_rows_1, n_clos);

  GPUtranspose(bpcache.d_a1, mat_t.d_a1_t, nn.n_rows_0, n_clos);

  cudaMemcpy(bpgrads.d_dW1, nn.d_W1, nn.n_rows_1 * nn.n_cols_1 * sizeof(real),
             cudaMemcpyDeviceToDevice);

  myGEMM(temp.d_diff, mat_t.d_a1_t, bpgrads.d_dW1, &alpha, &reg, nn.n_rows_1,
         nn.n_rows_0, n_clos);

  GPUcolumnSum(temp.d_diff, bpgrads.d_db1, nn.n_rows_1, n_clos);

  GPUtranspose(nn.d_W1, mat_t.d_W1_t, nn.n_rows_1, nn.n_cols_1);

  myGEMM(mat_t.d_W1_t, temp.d_diff, temp.d_da1, &alpha, &beta, nn.n_cols_1,
         n_clos, nn.n_rows_1);

  GPUhadamardProduct(temp.d_da1, bpcache.d_a1, temp.d_dz1_t1, nn.n_cols_1, n_clos);
  GPUhadamardProduct(temp.d_dz1_t1, bpcache.d_a1, temp.d_dz1_t2, nn.n_cols_1, n_clos);
  GPUadd(temp.d_dz1_t1, temp.d_dz1_t2, temp.d_dz1, 1.0, -1.0, nn.n_cols_1, n_clos);

  cudaMemcpy(bpgrads.d_dW0, nn.d_W0, nn.n_rows_0 * nn.n_cols_0 * sizeof(real),
             cudaMemcpyDeviceToDevice);

  GPUtranspose(d_x_batch, mat_t.d_X_batch_t, nn.n_cols_0, n_clos);

  myGEMM(temp.d_dz1, mat_t.d_X_batch_t, bpgrads.d_dW0, &alpha, &reg, nn.n_rows_0,
         nn.n_cols_0, n_clos);

  GPUcolumnSum(temp.d_dz1, bpgrads.d_db0, nn.n_rows_0, n_clos);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
real loss(NeuralNetwork &nn, const arma::Mat<real> &yc,
          const arma::Mat<real> &y, real reg)
{
  int N = yc.n_cols;
  real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  real data_loss = ce_sum / N;
  real reg_loss = 0.5 * reg * norms(nn);
  real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork &nn, const arma::Mat<real> &X,
             arma::Row<real> &label)
{
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i)
  {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork &nn, const arma::Mat<real> &X,
             const arma::Mat<real> &y, real reg, struct grads &numgrads)
{
  real h = 0.00001;
  struct cache numcache;
  numgrads.dW.resize(nn.num_layers);
  numgrads.db.resize(nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i)
  {
    numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

    for (int j = 0; j < nn.W[i].n_rows; ++j)
    {
      for (int k = 0; k < nn.W[i].n_cols; ++k)
      {
        real oldval = nn.W[i](j, k);
        nn.W[i](j, k) = oldval + h;
        feedforward(nn, X, numcache);
        real fxph = loss(nn, numcache.yc, y, reg);
        nn.W[i](j, k) = oldval - h;
        feedforward(nn, X, numcache);
        real fxnh = loss(nn, numcache.yc, y, reg);
        numgrads.dW[i](j, k) = (fxph - fxnh) / (2 * h);
        nn.W[i](j, k) = oldval;
      }
    }
  }

  for (int i = 0; i < nn.num_layers; ++i)
  {
    numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

    for (int j = 0; j < nn.b[i].size(); ++j)
    {
      real oldval = nn.b[i](j);
      nn.b[i](j) = oldval + h;
      feedforward(nn, X, numcache);
      real fxph = loss(nn, numcache.yc, y, reg);
      nn.b[i](j) = oldval - h;
      feedforward(nn, X, numcache);
      real fxnh = loss(nn, numcache.yc, y, reg);
      numgrads.db[i](j) = (fxph - fxnh) / (2 * h);
      nn.b[i](j) = oldval;
    }
  }
}

/*
 * Train the neural network nn
 */
void train(NeuralNetwork &nn, const arma::Mat<real> &X,
           const arma::Mat<real> &y, real learning_rate, real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug)
{
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch)
    {
      int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      arma::Mat<real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<real> y_batch = y.cols(batch * batch_size, last_col);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0)
      {
        if (grad_check)
        {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i)
      {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i)
      {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to CPUmats folder. In the later runs (with same parameters),
         you can use just the debug flag to
         output diff b/w CPU and GPU without running CPU version */
      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag)
      {
        write_cpudata_tofile(nn, iter);
      }

      iter++;
    }
  }
}

/*
 * TODO
 * Train the neural network nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork &nn, const arma::Mat<real> &X,
                    const arma::Mat<real> &y, real learning_rate, real reg,
                    const int epochs, const int batch_size, bool grad_check,
                    int print_every, int debug)
{
  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int N = X.n_cols;
  reg = reg / num_procs;

  std::ofstream error_file;
  error_file.open("Outputs/CpuGpuDiff.txt");
  int print_flag = 0;

  int sub_batch_size = (batch_size + num_procs - 1) / num_procs;
  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way. Or you can allocate your own
     array memory space and store the elements in a row major way. Remember to
     update the Armadillo matrices in NeuralNetwork &nn of rank 0 before
     returning from the function. */

  // Set up weights and biases of NN on GPU
  GPU_NN gpu_nn;
  cudaMalloc(&gpu_nn.d_W0, nn.W[0].n_rows * nn.W[0].n_cols * sizeof(real));
  cudaMalloc(&gpu_nn.d_W1, nn.W[1].n_rows * nn.W[1].n_cols * sizeof(real));
  cudaMalloc(&gpu_nn.d_b0, nn.b[0].n_rows * sizeof(real));
  cudaMalloc(&gpu_nn.d_b1, nn.b[1].n_rows * sizeof(real));

  cudaMemcpy(gpu_nn.d_W0, nn.W[0].memptr(),
             nn.W[0].n_rows * nn.W[0].n_cols * sizeof(real),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_nn.d_W1, nn.W[1].memptr(),
             nn.W[1].n_rows * nn.W[1].n_cols * sizeof(real),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_nn.d_b0, nn.b[0].memptr(),
             nn.b[0].n_rows * sizeof(real),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_nn.d_b1, nn.b[1].memptr(),
             nn.b[1].n_rows * sizeof(real),
             cudaMemcpyHostToDevice);

  gpu_nn.n_cols_0 = nn.W[0].n_cols;
  gpu_nn.n_rows_0 = nn.W[0].n_rows;
  gpu_nn.n_cols_1 = nn.W[1].n_cols;
  gpu_nn.n_rows_1 = nn.W[1].n_rows;

  GPU_cache gpu_bpcache;
  cudaMalloc(&gpu_bpcache.d_z1, nn.W[0].n_rows * sub_batch_size * sizeof(real));
  cudaMalloc(&gpu_bpcache.d_a1, nn.W[0].n_rows * sub_batch_size * sizeof(real));
  cudaMalloc(&gpu_bpcache.d_z2, nn.W[1].n_rows * sub_batch_size * sizeof(real));
  cudaMalloc(&gpu_bpcache.d_yc, nn.W[1].n_rows * sub_batch_size * sizeof(real));

  GPU_grads gpu_bpgrads;
  cudaMalloc(&gpu_bpgrads.d_dW0, nn.W[0].n_rows * nn.W[0].n_cols * sizeof(real));
  cudaMalloc(&gpu_bpgrads.d_dW1, nn.W[1].n_rows * nn.W[1].n_cols * sizeof(real));
  cudaMalloc(&gpu_bpgrads.d_db0, nn.b[0].n_rows * sizeof(real));
  cudaMalloc(&gpu_bpgrads.d_db1, nn.b[1].n_rows * sizeof(real));

  BP_temp bp_temp;
  cudaMalloc(&bp_temp.d_diff, nn.W[1].n_rows * sub_batch_size * sizeof(real));
  cudaMalloc(&bp_temp.d_da1, nn.W[0].n_rows * sub_batch_size * sizeof(real));
  cudaMalloc(&bp_temp.d_dz1, nn.W[0].n_rows * sub_batch_size * sizeof(real));
  cudaMalloc(&bp_temp.d_dz1_t1, nn.W[0].n_rows * sub_batch_size * sizeof(real));
  cudaMalloc(&bp_temp.d_dz1_t2, nn.W[0].n_rows * sub_batch_size * sizeof(real));

  Matrix_t matrix_t;
  cudaMalloc(&matrix_t.d_a1_t, sub_batch_size * nn.W[0].n_rows * sizeof(real));
  cudaMalloc(&matrix_t.d_W1_t, nn.W[1].n_cols * nn.W[1].n_rows * sizeof(real));
  cudaMalloc(&matrix_t.d_X_batch_t, sub_batch_size * nn.W[0].n_cols * sizeof(real));

  real *d_X_batch, *d_y_batch;
  cudaMalloc(&d_X_batch, nn.W[0].n_cols * sub_batch_size * sizeof(real));
  cudaMalloc(&d_y_batch, nn.W[1].n_rows * sub_batch_size * sizeof(real));

  GPU_grads cpu_bpgrads;
  cpu_bpgrads.d_dW0 = (real *)malloc(nn.W[0].n_rows * nn.W[0].n_cols * sizeof(real));
  cpu_bpgrads.d_dW1 = (real *)malloc(nn.W[1].n_rows * nn.W[1].n_cols * sizeof(real));
  cpu_bpgrads.d_db0 = (real *)malloc(nn.b[0].n_rows * sizeof(real));
  cpu_bpgrads.d_db1 = (real *)malloc(nn.b[1].n_rows * sizeof(real));

  int num_batches = (N + batch_size - 1) / batch_size;
  int batch_clos = batch_size;

  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;
  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    for (int batch = 0; batch < num_batches; ++batch)
    {
      /*
       * Possible implementation:
       * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
       * 2. compute each sub-batch of images' contribution to network
       * coefficient updates
       * 3. reduce the coefficient updates and broadcast to all nodes with
       * `MPI_Allreduce()'
       * 4. update local network coefficient at each node
       */

      int start_idx = batch * batch_size;
      int end_idx = std::min((batch + 1) * batch_size - 1, N - 1);
      batch_clos = end_idx - start_idx + 1;

      int sub_start_idx = start_idx + rank * sub_batch_size;
      int sub_end_idx = std::min(start_idx + (rank + 1) * sub_batch_size - 1,
                                 end_idx);

      if (sub_start_idx <= sub_end_idx)
      {
        arma::Mat<real> sub_X_batch = X.cols(sub_start_idx, sub_end_idx);
        arma::Mat<real> sub_y_batch = y.cols(sub_start_idx, sub_end_idx);

        int sub_X_batch_n_cols = sub_X_batch.n_cols;
        int sub_X_batch_n_rows = sub_X_batch.n_rows;
        int sub_y_batch_n_cols = sub_y_batch.n_cols;
        int sub_y_batch_n_rows = sub_y_batch.n_rows;

        cudaMemcpy(d_X_batch, sub_X_batch.memptr(),
                   sub_X_batch_n_cols * sub_X_batch_n_rows * sizeof(real),
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_y_batch, sub_y_batch.memptr(),
                   sub_y_batch_n_cols * sub_y_batch_n_rows * sizeof(real),
                   cudaMemcpyHostToDevice);

        GPUfeedforward(gpu_nn, gpu_bpcache, d_X_batch, sub_X_batch_n_cols);

        GPUbackprop(gpu_nn, reg, gpu_bpcache, gpu_bpgrads, bp_temp, matrix_t,
                    d_X_batch, d_y_batch, sub_y_batch_n_cols, batch_clos);

        // Transfer gradients to CPU
        cudaMemcpy(cpu_bpgrads.d_dW0, gpu_bpgrads.d_dW0,
                   nn.W[0].n_rows * nn.W[0].n_cols * sizeof(real),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(cpu_bpgrads.d_dW1, gpu_bpgrads.d_dW1,
                   nn.W[1].n_rows * nn.W[1].n_cols * sizeof(real),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(cpu_bpgrads.d_db0, gpu_bpgrads.d_db0,
                   nn.b[0].n_rows * sizeof(real),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(cpu_bpgrads.d_db1, gpu_bpgrads.d_db1,
                   nn.b[1].n_rows * sizeof(real),
                   cudaMemcpyDeviceToHost);
      }
      else
      {
        memset(cpu_bpgrads.d_dW0, 0, nn.W[0].n_rows * nn.W[0].n_cols * sizeof(real));
        memset(cpu_bpgrads.d_dW1, 0, nn.W[1].n_rows * nn.W[1].n_cols * sizeof(real));
        memset(cpu_bpgrads.d_db0, 0, nn.b[0].n_rows * sizeof(real));
        memset(cpu_bpgrads.d_db1, 0, nn.b[1].n_rows * sizeof(real));
      }

      // Reduce gradients
      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, cpu_bpgrads.d_dW0,
                                  nn.W[0].n_rows * nn.W[0].n_cols, MPI_FP,
                                  MPI_SUM, MPI_COMM_WORLD));

      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, cpu_bpgrads.d_dW1,
                                  nn.W[1].n_rows * nn.W[1].n_cols, MPI_FP,
                                  MPI_SUM, MPI_COMM_WORLD));

      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, cpu_bpgrads.d_db0,
                                  nn.b[0].n_rows, MPI_FP,
                                  MPI_SUM, MPI_COMM_WORLD));

      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, cpu_bpgrads.d_db1,
                                  nn.b[1].n_rows, MPI_FP,
                                  MPI_SUM, MPI_COMM_WORLD));

      // Copy gradients back to GPU
      cudaMemcpy(gpu_bpgrads.d_dW0, cpu_bpgrads.d_dW0,
                 nn.W[0].n_rows * nn.W[0].n_cols * sizeof(real),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(gpu_bpgrads.d_dW1, cpu_bpgrads.d_dW1,
                 nn.W[1].n_rows * nn.W[1].n_cols * sizeof(real),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(gpu_bpgrads.d_db0, cpu_bpgrads.d_db0,
                 nn.b[0].n_rows * sizeof(real),
                 cudaMemcpyHostToDevice);

      cudaMemcpy(gpu_bpgrads.d_db1, cpu_bpgrads.d_db1,
                 nn.b[1].n_rows * sizeof(real),
                 cudaMemcpyHostToDevice);

      // Gradient descent step
      GPUadd(gpu_nn.d_W0, gpu_bpgrads.d_dW0, gpu_nn.d_W0, 1.0, -learning_rate,
             nn.W[0].n_rows, nn.W[0].n_cols);

      GPUadd(gpu_nn.d_W1, gpu_bpgrads.d_dW1, gpu_nn.d_W1, 1.0, -learning_rate,
             nn.W[1].n_rows, nn.W[1].n_cols);

      GPUadd(gpu_nn.d_b0, gpu_bpgrads.d_db0, gpu_nn.d_b0, 1.0, -learning_rate,
             nn.b[0].n_rows, 1);

      GPUadd(gpu_nn.d_b1, gpu_bpgrads.d_db1, gpu_nn.d_b1, 1.0, -learning_rate,
             nn.b[1].n_rows, 1);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      if (debug && rank == 0 && print_flag)
      {
        // TODO
        // Copy data back to the CPU
        cudaMemcpy(nn.W[0].memptr(), gpu_nn.d_W0,
                   nn.W[0].n_rows * nn.W[0].n_cols * sizeof(real),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(nn.W[1].memptr(), gpu_nn.d_W1,
                   nn.W[1].n_rows * nn.W[1].n_cols * sizeof(real),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(nn.b[0].memptr(), gpu_nn.d_b0,
                   nn.b[0].n_rows * sizeof(real),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(nn.b[1].memptr(), gpu_nn.d_b1,
                   nn.b[1].n_rows * sizeof(real),
                   cudaMemcpyDeviceToHost);

        /* The following debug routine assumes that you have already updated the
         arma matrices in the NeuralNetwork nn.  */
        write_diff_gpu_cpu(nn, iter, error_file);
      }

      iter++;
    }
  }

  // TODO
  // Copy data back to the CPU
  cudaMemcpy(nn.W[0].memptr(), gpu_nn.d_W0,
             nn.W[0].n_rows * nn.W[0].n_cols * sizeof(real),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(nn.W[1].memptr(), gpu_nn.d_W1,
             nn.W[1].n_rows * nn.W[1].n_cols * sizeof(real),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(nn.b[0].memptr(), gpu_nn.d_b0,
             nn.b[0].n_rows * sizeof(real),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(nn.b[1].memptr(), gpu_nn.d_b1,
             nn.b[1].n_rows * sizeof(real),
             cudaMemcpyDeviceToHost);
  error_file.close();

  // TODO
  // Free memory
  cudaFree(gpu_nn.d_W0);
  cudaFree(gpu_nn.d_W1);
  cudaFree(gpu_nn.d_b0);
  cudaFree(gpu_nn.d_b1);

  cudaFree(gpu_bpcache.d_z1);
  cudaFree(gpu_bpcache.d_a1);
  cudaFree(gpu_bpcache.d_z2);
  cudaFree(gpu_bpcache.d_yc);

  cudaFree(gpu_bpgrads.d_dW0);
  cudaFree(gpu_bpgrads.d_dW1);
  cudaFree(gpu_bpgrads.d_db0);
  cudaFree(gpu_bpgrads.d_db1);

  cudaFree(bp_temp.d_diff);
  cudaFree(bp_temp.d_da1);
  cudaFree(bp_temp.d_dz1);
  cudaFree(bp_temp.d_dz1_t1);
  cudaFree(bp_temp.d_dz1_t2);

  cudaFree(matrix_t.d_a1_t);
  cudaFree(matrix_t.d_W1_t);
  cudaFree(matrix_t.d_X_batch_t);

  cudaFree(d_X_batch);
  cudaFree(d_y_batch);

  free(cpu_bpgrads.d_dW0);
  free(cpu_bpgrads.d_dW1);
  free(cpu_bpgrads.d_db0);
  free(cpu_bpgrads.d_db1);
}