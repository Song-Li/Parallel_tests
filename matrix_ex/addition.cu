
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
//#include <chrono>
#include <time.h>
#include <cmath>
//#include <random>

using namespace std;

#define BLOCK_SIZE 16

//mt19937_64 rd;

typedef struct
{
	int height, width;
	float *elements;
}Matrix;

__global__ void MatAddKernel(const Matrix, const Matrix, Matrix);

bool check(const Matrix A, const Matrix B, const Matrix C)
{
	float eps = 1e-10;
	for(size_t i = 0; i < A.height * A.width; ++i)
	{
		if (fabs(A.elements[i] + B.elements[i] - C.elements[i]) > eps)
			return false;
	}
	return true;
}

void MatAdd(const Matrix A, const Matrix B, Matrix C)
{
	Matrix d_A, d_B, d_C;
	size_t total_size = sizeof(float) *	A.height * A.width;
	d_A.height = A.height; d_A.width = A.width;
	cudaError_t err = cudaMalloc(&d_A.elements, total_size);
	if (err != cudaSuccess)
	{
		printf("fail to malloc A: %s", cudaGetErrorString(err));
	}
	err = cudaMemcpy(d_A.elements, A.elements, total_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("fail to memcpy A: %s\n", cudaGetErrorString(err));
	}

	d_B.height = B.height; d_B.width = B.width;
	err = cudaMalloc(&d_B.elements, total_size);
	if (err != cudaSuccess)
	{
		printf("fail to malloc B: %s", cudaGetErrorString(err));
	}
	err = cudaMemcpy(d_B.elements, B.elements, total_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("fail to memcpy B: %s\n", cudaGetErrorString(err));
	}

	d_C.height = C.height; d_C.width = C.width;
	err = cudaMalloc(&d_C.elements, total_size);
	if (err != cudaSuccess)
	{
		printf("fail to malloc C: %s", cudaGetErrorString(err));
	}

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((A.width + dimBlock.x - 1) / dimBlock.x, 
		(A.height + dimBlock.y - 1) / dimBlock.y);

	// Time region begins
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	MatAddKernel<<<dimGrid, dimBlock>>> (d_A, d_B, d_C);
	cudaEventRecord(stop);

	cudaMemcpy(C.elements, d_C.elements, total_size, cudaMemcpyDeviceToHost);

	err = cudaEventSynchronize(stop);
	if (err == cudaSuccess)
	{
		printf("computing done\n");
	}
	else
	{
		printf("fail to compute\n");
	}

	// Time region ends
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("total time: %.4f ms\n", milliseconds);
}

__global__ void MatAddKernel(const Matrix A, const Matrix B, Matrix C)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < A.width && row < A.height)
		C.elements[row * C.width + col] = A.elements[row * A.width + col] + B.elements[row * B.width + col];
}


void generate_matrix(Matrix &A)
{
	size_t total = A.height * A.width;
	for(size_t i = 0; i < total; ++i)
	{
		A.elements[i] = rand();
	}
}

int main(int argc, char **argv)
{
	/* Usage:
		EXE HEIGHT WIDTH SEED
	*/
	if (argc < 4)
	{
		printf("Usage: %s HEIGHT WIDTH SEED\n", argv[0]);
		return 0;
	}
	
	// To parse arguments
	int height = atoi(argv[1]);
	int width = atoi(argv[2]);
	int seed = atol(argv[3]);
	
	// To seed the random device
	//rd.seed(seed);

	// To set up and gerenerate random matrix
	Matrix A, B, C;
	size_t matrix_size = sizeof(float) * height * width;
	A.height = height; A.width = width;
	A.elements = static_cast<float*>(malloc(matrix_size));
	generate_matrix(A);

	B.height = height; B.width = width;
	B.elements = static_cast<float*>(malloc(matrix_size));
	generate_matrix(B);

	C.height = height; C.width = width;
	C.elements = static_cast<float*>(malloc(matrix_size));

	printf("Matrices are generated.\n");

	MatAdd(A, B, C);

	if (check(A, B, C))
	{
		printf("check succeed\n");
	}
	else
	{
		printf("check failed\n");
	}

    return 0;
}

 
