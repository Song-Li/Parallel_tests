#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#define MAX 8192
#define n_threads 8
#define nD 1024
#define K 50
#define MaxIter 1500

typedef struct{
    float data[nD];
    int center;
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
    Point *res = (Point *)malloc(size * sizeof(Point));
    for(int i = 0;i < size;++ i){
        for(int j = 0;j < nD;++ j)
            res[i].data[j] = i;//(float)((float)rand() / RAND_MAX) * 1024.0;
        res[i].center = -1;
    }
    return res;
}

void *do_step(void *argus){
    Args *args = (Args *)argus;
    Point *array = args->array, *center = args->center;
    int start = args->start, end = args->end, n_k = args->k;
    int tmp_center;
    float x_i, y_i, dist, min_dist , tmp_float;
    for(int i = start;i < end;++ i){
        min_dist = 0xffffff3f3f;
        tmp_center = array[i].center;
        for(int j = 0;j < n_k;++ j){
            dist = 0;
            for(int k = 0;k < nD;++ k){
                tmp_float = array[i].data[k] - center[j].data[k];
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
    pthread_barrier_wait(&barr);
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
    pthread_barrier_init(&barr, NULL, n_threads);
    Args args[n_threads];// = (Args *)malloc(sizeof(Args) * n_threads);

    //outputPoint(array, size);

    for(int i = 0;i < n_threads;++ i){
        args[i].k = K;
        args[i].array = array;
        args[i].center = centers;
        args[i].start = i * block_size;
        if(i != n_threads - 1) {
            args[i].end = args[i].start + block_size;
        }
        else {
            args[i].end = size;
        }
    }

    int iter = MaxIter;
    auto starttime = std::chrono::high_resolution_clock::now();
    while(iter --){
        finished = 1;
        for(int i = 0;i < n_threads;++ i){
            if(i != n_threads - 1) 
                pthread_create(&threads[i], NULL, do_step, &args[i]);
            else 
                do_step(&args[i]);
        }

        for(int i = 0;i < n_threads - 1;++ i)
            pthread_join(threads[i], NULL);

        memset(centers, 0, sizeof(Point) * K);
        int index = 0;
        for(int i = 0;i < size;++ i){
            index = array[i].center;
            centers[index].center ++;
            for(int j = 0;j < nD;++ j)
                centers[index].data[j] += array[i].data[j];
        }

        for(int i = 0; i < K;++ i){
            for(int j = 0;j < nD;++ j){
                centers[i].data[j] /= centers[i].center;
            }
        }

    }
    auto endtime = std::chrono::high_resolution_clock::now();
    std::cout << "Run Time: " << (double)std::chrono::duration_cast<std::chrono::microseconds>(endtime - starttime).count()/1000000 << std::endl;
    printf("Iters: %d\n", iter);
    return 0;
}
