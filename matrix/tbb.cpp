#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/task_scheduler_init.h"
#define MAX 4096

float* getNew(ssize_t g_size) {
    return (float *)calloc(g_size * g_size, sizeof(float));
}

float *mul(float *mat_a, float *mat_b, int size){
    float * result = getNew(size);
    float now;
    tbb::parallel_for (tbb::blocked_range<int>(0, size, 32),[&] (const tbb::blocked_range<int>& r) {
        for(int i = r.begin();i != r.end();++ i){
            int inbase = i * size;
            for(int j = 0;j < size;++ j){
                int base = j * size;
                now = mat_a[i * size + j];
                for(int k = 0;k < size;++ k){
                    result[inbase + k] += now * mat_b[base + k];
                }
            }
        }
    });
    return result;
}

int main(){
    float *mat_a;
    float *mat_b;
    mat_a = (float *)malloc(sizeof(float) * MAX * MAX);
    mat_b = (float *)malloc(sizeof(float) * MAX * MAX);
    for(int i = 0;i < MAX * MAX;++ i) {
        mat_a[i] = (float)rand()/(float)(RAND_MAX/20);
        mat_b[i] = (float)rand()/(float)(RAND_MAX/20);
    }
    int res = 0;
    auto starttime = std::chrono::high_resolution_clock::now();
    float *result = mul(mat_a, mat_b, MAX);
    auto endtime = std::chrono::high_resolution_clock::now();
    std::cout << (double)std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count()/1000000 << std::endl;
    printf("%f\n", result[0]);
    return 0;

}
