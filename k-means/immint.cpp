#include <stdio.h>
#include <immintrin.h>
#include "tbb/parallel_for.h"
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#define MAX 8192
#define n_threads 8
#define nD 1024
#define K 50
#define MaxIter 1500

typedef struct{
    float data[nD];
    long long center;
    long long occu[3];
}Point;

typedef struct{
    Point *array, *center;
    int start, end, size, k;
}Args;


pthread_barrier_t barr;
int finished = 0;

Point *getNew(int size){
    return (Point *)calloc(size, sizeof(Point));
}

Point *getRandom(int size){
    Point *res = (Point *)malloc(size * sizeof(Point) + 31);
    res = reinterpret_cast<Point *>((reinterpret_cast<uintptr_t>(res) + 31) / 32 * 32);
    for(int i = 0;i < size;++ i){
        for(int j = 0;j < nD;++ j)
            res[i].data[j] = i;//(float)((float)rand() / RAND_MAX) * 1024.0;
        res[i].center = -1;
    }
    return res;
}

void do_step(Point *array, Point *center){
    const size_t size = MAX;
    Point *__restrict__ da = array;
    Point *__restrict__ const dc = center;

    const size_t chunk_size = 8;
    const size_t chunks = nD / chunk_size;

#pragma omp parallel for num_threads(8)
    for(int i = 0;i < size;++ i){

        __m256 a_line, c_line, r_line;
        float min_dist = 0xffffff3f3f, tmp_float;
        int tmp_center = array[i].center;
        float dist = 0;
        for(int j = 0;j < K;++ j){
            dist = 0;
            for(int k = 0;k <= chunks - chunk_size;k += chunk_size){
                a_line = _mm256_load_ps(&array[i].data[k]);
                c_line = _mm256_load_ps(&center[j].data[k]);
                r_line = _mm256_sub_ps(a_line, c_line);
                r_line = _mm256_mul_ps(r_line, r_line);
                __m256 hsum = _mm256_hadd_ps(r_line, r_line);

                hsum = _mm256_add_ps(hsum, _mm256_permute2f128_ps(hsum, hsum, 0x1));

                _mm_store_ss(&tmp_float, _mm_hadd_ps( _mm256_castps256_ps128(hsum), _mm256_castps256_ps128(hsum) ) );
                dist += tmp_float;
            }

            for(int k = chunk_size * chunks; k < nD; ++ k){
                tmp_float = array[i].data[k] - center[i].data[k];
                dist += tmp_float * tmp_float;
            }

            if(dist < min_dist){
                min_dist = dist;
                tmp_center = j;
            }
        }
        if(tmp_center != array[i].center){
            finished = 0;
            array[i].center = tmp_center;
        }
    }
}

void outputPoint(Point *array, int size){
    for(int i = 0;i < size;++ i){
        for(int j = 0;j < nD;++ j){
            printf("%f ", array[i].data[j]);
        }
        printf("\n");
    }
}

int main(){
    const int size = MAX;
    Point *array = getRandom(size);
    Point *centers = getRandom(K);
    memcpy(centers, array, sizeof(Point) * K);
    Point *result = getNew(size);
    pthread_t threads[n_threads];
    int block_size = size / n_threads;
    Args args[n_threads];

    int iter = MaxIter;
    auto starttime = std::chrono::high_resolution_clock::now();
    while(iter --){
        finished = 1;
        do_step(array, centers);

        memset(centers, 0, sizeof(Point) * K);
        int index = 0;
        for(int i = 0;i < size;++ i){
            index = array[i].center;
            if(index < 0) printf("%d %d\n", i, index);
            centers[index].center ++;
            for(int j = 0;j < nD;++ j)
                centers[index].data[j] += array[i].data[j];
        }
        for(int i = 0;i < K;++ i){
            for(int j = 0;j < nD;++ j)
                if(centers[i].center){ 
                    centers[i].data[j] /= centers[i].center;
                }
        }
    }
    auto endtime = std::chrono::high_resolution_clock::now();
    std::cout << "Run Time:" << (double)std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count()/1000000 << std::endl;
    printf("Iters: %d\n", iter);
    return 0;
}
