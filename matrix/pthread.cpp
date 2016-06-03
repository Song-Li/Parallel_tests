#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <pthread.h>
#include <iostream>
#include <chrono>

const ssize_t g_size = 4096;
#define min(a,b) (a) < (b) ? (a) : (b)

typedef struct
{
    const float *a, *b;
    float *c;
    ssize_t row_b, row_e;
}Args;

void* _matrix_mul(void *argv)
{
    Args* multi = (Args*)(argv);
    const float *matrix_a = multi->a, *matrix_b = multi->b;
    float *result = multi->c;
    const ssize_t row_b = multi->row_b, row_e = multi->row_e;

    // (i, k, j) order has better cache performance
    for (ssize_t i = row_b; i < row_e; i++)
    {
        const ssize_t base = i * g_size;
        for (ssize_t k = 0; k < g_size; k++)
        {
            float c = matrix_a[base + k];
            if (c == 0.0)
                continue;
            const size_t inbase = k * g_size;
            for (ssize_t j = 0; j < g_size; j++)
            {
                result[base + j] += c * matrix_b[inbase + j];
            }
        }
    }

    return NULL;
}
float* new_matrix(void) {
    return (float *)calloc(g_size * g_size, sizeof(float));
}
pthread_t threads[16];

float* matrix_mul(const float* matrix_a, const float* matrix_b, ssize_t g_size)
{
    float *result = new_matrix();

    // To make sure the overhead of creating threads is small
    const ssize_t block_size = 128;
    int total_threads = min(16, (g_size + block_size - 1) / block_size);
    total_threads = 8;
    ssize_t per_block = g_size / total_threads;

    Args multis[16];
    ssize_t now_row = 0;
    ssize_t end_row = g_size;

    // Split the matrix A by row and make multiplication parallel
    for(int i = 0; i < total_threads - 1; i++)
    {
        multis[i].a = matrix_a; multis[i].b = matrix_b;
        multis[i].c = result;
        multis[i].row_b = now_row;
        multis[i].row_e = now_row + per_block;
        pthread_create(&threads[i], NULL, _matrix_mul, &multis[i]);
        now_row += per_block;
    }

    // Don't let main thread free
    for (ssize_t i = now_row; i < end_row; i++)
    {
        const ssize_t base = i * g_size;
        for (ssize_t k = 0; k < g_size; k++)
        {
            float c = matrix_a[base + k];
            if (c == 0.0)
                continue;
            const ssize_t inbase = k * g_size;
            for (ssize_t j = 0; j < g_size; j++)
            {
                result[base + j] += c * matrix_b[inbase + j];
            }
        }
    }

    for(int i = 0; i < total_threads - 1; i++)
    {
        pthread_join(threads[i], NULL);
    }

    return result;
}

int main(){
    float *mat_a;
    float *mat_b;
    float res = 0;
    mat_a = (float *)malloc(sizeof(float) * g_size * g_size);
    mat_b = (float *)malloc(sizeof(float) * g_size * g_size);
    for(int i = 0;i < g_size * g_size;++ i) {
        mat_a[i] = (float)rand()/(float)(RAND_MAX/20);
        mat_b[i] = (float)rand()/(float)(RAND_MAX/20);
    }
    auto starttime = std::chrono::high_resolution_clock::now();
    float *result = matrix_mul(mat_a, mat_b, g_size);
    auto endtime = std::chrono::high_resolution_clock::now();
    std::cout << (double)std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count()/1000000 << std::endl;
    printf("%f\n", result[0]);
    return 0;
}
