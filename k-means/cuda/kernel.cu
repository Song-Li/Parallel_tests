
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>

using namespace std;

int num_samples;
int sample_dimension;
int num_clusters;


typedef struct
{
	int width, height;
	float *elements;
}Matrix;

typedef struct
{
	int height;
	int *elements;
}Label;

typedef struct
{
	int num_loops;
	Label labels;
	Matrix cluster_centers;
}KmeansRecord;

__global__ void KmeansKernelFindClusters(Matrix, Matrix, Label, int *);
__global__ void kmeansKernelFindCenters(Matrix, Label);

Matrix createMatrix(const int height, const int width)
{
	Matrix mat;
	mat.width = width;
	mat.height = height;
	mat.elements = (float*)calloc(1ul * width * height, sizeof(float));
	return mat;
}

void freeMatrix(Matrix mat)
{
	free(mat.elements);
}

void freeLabel(Label lab)
{
	free(lab.elements);
}

Matrix cloneMatrix(const Matrix &mat)
{
	Matrix new_mat = createMatrix(mat.height, mat.width);
	memcpy(new_mat.elements, mat.elements, sizeof(float) * new_mat.height * new_mat.width);
	return new_mat;
}

__global__ void kmeansKernelFindCenters(Matrix samples, Label labels, Matrix cluster_centers)
{
	int term = blockIdx.y * blockDim.y + threadIdx.y, term_count = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int n = samples.height, m = samples.width, k = cluster_centers.height;
	float val = 0.0;

	if (term >= k || col >= m)
		return;

	for (int i = 0; i < n; ++i)
	{
		int f = (labels.elements[i] == term);
		term_count += f;
		val += samples.elements[i * m + col] * f;
	}

	term_count = (term_count) ? term_count : 1;
	cluster_centers.elements[term * m + col] = val / term_count;
}

__host__ __device__ float euclid_dist_square(Matrix samples, Matrix cluster_centers
											, int sample, int cluster)
{
	float ret = 0.0;
	int num_samples = samples.height;
	int sample_dimension = samples.width;

	for (int i = 0; i < sample_dimension; ++i)
	{
		float current = samples.elements[sample_dimension * sample + i] - cluster_centers.elements[sample_dimension * cluster + i];
		current *= current;
		ret += current;
	}

	return ret;
}


__global__ void KmeansKernelFindClusters(Matrix samples, Matrix cluster_centers, Label new_labels)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int num_clusters = cluster_centers.height;
	
	if (row < samples.height) 
	{
		int best = -1;
		float dist, min_dist;

		best = 0;
		min_dist = euclid_dist_square(samples, cluster_centers, row, 0);

		for(int i = 1; i < num_clusters; ++i)
		{
			dist = euclid_dist_square(samples, cluster_centers, row, i);
			if (dist < min_dist)
			{
				min_dist = dist;
				best = i;
			}
		}
		new_labels.elements[row] = best;
	}
}

float KmeansCalcualteSSE(Matrix samples, Matrix cluster_centers, Label labels)
{
	int n = samples.height, m = samples.width, k = labels.height;
	
	float sse = 0.0;

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			int label = labels.elements[i];
			float diff = samples.elements[i * m + j] - cluster_centers.elements[label * m + j];
			sse += diff * diff;
		}
	}
	return sse;
}

KmeansRecord do_kmeans(const Matrix samples, const Matrix initial_centers, const int max_iter=1500)
{
	const int block_size = 16;
	int n = samples.height, m = samples.width, k = initial_centers.height;
	// To prepare the calculation
	Label labels;
	labels.height = n;
	size_t size = sizeof(float) * n;
	labels.elements = (int*)malloc(size);

	Label d_labels;
	d_labels.height = n;
	size = sizeof(float) * n;
	cudaError_t err = cudaMalloc(&d_labels.elements, size);

	// To copy samples and initial_centers to the device
	Matrix d_samples;
	d_samples.height = n; d_samples.width = m;
	size = sizeof(float) * n * m;
	err = cudaMalloc(&d_samples.elements, size);
	if (err)
		printf("fail to malloc samples: %s\n", cudaGetErrorString(err));
	err  = cudaMemcpy(d_samples.elements, samples.elements, size, cudaMemcpyHostToDevice);
	if (err)
		printf("fail to copy samples to device: %s\n", cudaGetErrorString(err));

	Matrix centers;
	centers.height = k; centers.width = m;
	size = sizeof(float) * k * m;
	centers.elements = (float*)malloc(size);

	Matrix d_centers;
	d_centers.height = k; d_centers.width = m;
	size = sizeof(float) * k * m;
	err = cudaMalloc(&d_centers.elements, size);
	if (err)
		printf("fail to malloc centers: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(d_centers.elements, initial_centers.elements, size, cudaMemcpyHostToDevice);
	if (err)
		printf("fail to copy centers to device: %s\n", cudaGetErrorString(err));

	dim3 dimBlock1(block_size * block_size);
	dim3 dimGrid1((n + block_size * block_size - 1) / (block_size * block_size));

	dim3 dimBlock2(block_size, block_size);
	dim3 dimGrid2((m + block_size - 1) / block_size, (k + block_size - 1) / block_size);
	
	KmeansRecord record;
	record.cluster_centers.height = k;
	record.cluster_centers.width = m;
	record.cluster_centers.elements = (float*)malloc(sizeof(float) * k * m);
	record.labels.height = n;
	record.labels.elements = (int*)malloc(sizeof(int) * n);

	float milliseconds = 0.0;
	float before_sse = -1.0;
	float this_sse = 0.0;

	int total_loops = 0;
	for (int i = 0; i < max_iter; ++i)
	{
		total_loops = i + 1;
		cudaEvent_t start, stop;
		float this_time = 0.0;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		KmeansKernelFindClusters<<<dimGrid1, dimBlock1>>>(d_samples, d_centers, d_labels);
		cudaEventRecord(stop);

		err = cudaEventSynchronize(stop);
		cudaEventElapsedTime(&this_time, start, stop);
		if (!err)
		{
			// printf("iteration %d: staget 1 done. Time: %.2f ms.\n ", i + 1, this_time);
			milliseconds += this_time;
		}

		cudaEventRecord(start);
		kmeansKernelFindCenters<<<dimGrid2, dimBlock2>>>(d_samples, d_labels, d_centers);
		cudaEventRecord(stop);

		err = cudaEventSynchronize(stop);
		cudaEventElapsedTime(&this_time, start, stop);
		if (!err)
		{
			// printf("iteration %d: staget 2 done. Time: %.2f ms.\n ", i + 1, this_time);
			milliseconds += this_time;
		}

		cudaMemcpy(labels.elements, d_labels.elements, sizeof(int) * n, cudaMemcpyDeviceToHost);
		cudaMemcpy(centers.elements, d_centers.elements, sizeof(float) * k * m, cudaMemcpyDeviceToHost); 
		this_sse = KmeansCalcualteSSE(samples, centers, labels);

		if (!err)
		{
			/*
			 printf("SSE: %.2f\n", this_sse);
			 if (i > 0 && fabs(this_sse - before_sse) == 0)
			 {
				printf("%d %.2f\n", i + 1, this_sse);
				break;
			}
			*/
			before_sse = this_sse;
		}
		else
		{
			printf("%s\n", cudaGetErrorString(err));
		}
	}

	printf("total iteration: %d; total time: %.2f ms\n", total_loops, milliseconds);

	memcpy(record.labels.elements, labels.elements, sizeof(int) * n);
	memcpy(record.cluster_centers.elements, centers.elements, sizeof(float) * m * k);

	freeMatrix(centers);
	freeLabel(labels);

	cudaFree(d_samples.elements);
	cudaFree(d_centers.elements);
	cudaFree(d_labels.elements);

	return record;
}


Matrix read_samples(const char *path, int *k)
{
	FILE* f = fopen(path, "rb");
	int n, m;
	fread(&n, 4, 1, f);
	fread(&m, 4, 1, f);
	fread(k, 4, 1, f);

	Matrix samples = createMatrix(n, m); 
	fread(samples.elements, 4, 1ul * n * m, f);

	return samples;
}

void save_record(KmeansRecord record)
{




}

int main(int argc, char **argv)
{
    /*
	if (argc < 2)
	{
		printf("usage: %s SOURCE\n", argv[0]);
		return 0;
	}
    */

	// To read data
	int k;
	// Matrix samples = read_samples(argv[1], &k);

	
	Matrix samples;
	samples.height = 8192;
	samples.width = 1024;
	samples.elements = (float*)malloc(sizeof(float) * 8192 * 1024);
	k = 50;

	for (int i = 0; i < 8192; ++i)
	{
		for(int j = 0; j < 1024; ++j)
		{
			samples.elements[i * 1024 + j] = i;
		}
	}
	


	Matrix initial_center = createMatrix(k, samples.width);

	memcpy(initial_center.elements, samples.elements, sizeof(float) * k);

	KmeansRecord record = do_kmeans(samples, initial_center);

	save_record(record);

	freeMatrix(samples);
	freeMatrix(initial_center);

	return 0;
}
