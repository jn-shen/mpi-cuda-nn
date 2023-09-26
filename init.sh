#!/bin/bash

TRAIN_IMG_FILE_NAME="train-images-idx3-ubyte.gz"
TRAIN_LABELS_FILE_NAME="train-labels-idx1-ubyte.gz"

TEST_IMG_FILE_NAME="t10k-images-idx3-ubyte.gz"

rm -rf data/
mkdir data
cd data

echo "Downloading training images..."
wget -O "$TRAIN_IMG_FILE_NAME" "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
gunzip "$TRAIN_IMG_FILE_NAME"

echo "Downloading training labels..."
wget -O "$TRAIN_LABELS_FILE_NAME" "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
gunzip "$TRAIN_LABELS_FILE_NAME"

echo "Downloading test images..."
wget -O "$TEST_IMG_FILE_NAME" "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
gunzip "$TEST_IMG_FILE_NAME"

rm *.gz
cd ..

module load gcc
module load cmake
echo "Downloading Lapack Library..."
wget http://www.netlib.org/lapack/lapack-3.8.0.tar.gz
tar xvfz lapack-3.8.0.tar.gz
cd lapack-3.8.0
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=../../lib/lapack ..
make
make install
cd ../..
rm -rf lapack-3.8.0*

export LD_LIBRARY_PATH=$PWD/lib/lapack/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$PWD/lib/lapack:$CMAKE_PREFIX_PATH

echo "Downloading Armadillo Library..."
wget http://sourceforge.net/projects/arma/files/armadillo-9.800.2.tar.xz
tar xvfJ armadillo-9.800.2.tar.xz
cd armadillo-9.800.2
./configure
make
make install DESTDIR=../lib/armadillo
cd ..
rm -rf armadillo-9.800.2*

mkdir -p obj

mkdir -p Outputs/CPUmats
