#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <immintrin.h>

using namespace std;

class matrix_t
{
    size_t _size;
    float *_data;
    float *_data_ori;
    public:
    float **data;

    matrix_t(size_t size):
        _size(size), _data_ori(static_cast<float*>(malloc(size * size * 8 + 31)))
        , data(new float*[size])
    {
        _data = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(_data_ori) + 31) / 32 * 32);
        memset(_data, 0, 8 * size * size);
        for(size_t i = 0; i < _size; ++i)
        {
            data[i] = &_data[i * size];
        }
    }

    ~matrix_t()
    {
        free(_data_ori);
        delete [] data;
    }

    size_t size() const
    {
        return _size;
    }
};

void generate_matrix(matrix_t &A, matrix_t &B, unsigned int seed)
{
    srand(seed);
    size_t n = A.size();
    float **da = A.data, **db = B.data;

    mt19937 rd(seed);
    normal_distribution<> norm;


    for(size_t i = 0; i < n; ++i)
    {
        for(size_t j = 0; j < n; ++j)
        {
            da[i][j] = norm(rd);
            db[i][j] = norm(rd);
        }
    }
}

void multi1(const matrix_t &A, const matrix_t &B, matrix_t &C)
{
    size_t n = A.size(); 
    float **__restrict__ const da = A.data;
    float **__restrict__ const db = B.data;
    float **__restrict__ dc = C.data;

#pragma omp parallel for num_threads(8)
    for(size_t i = 0; i < n; ++i)
    {
        for(size_t k = 0; k < n; ++k)
        {
            float c = da[i][k];
            for(size_t j = 0; j < n; ++j)
                dc[i][j] += c * db[k][j];
        }
    }

}

void multi2(const matrix_t &A, const matrix_t &B, matrix_t &C)
{
    size_t n = A.size(); 
    float **__restrict__ const da = A.data;
    float **__restrict__ const db = B.data;
    float **__restrict__ dc = C.data;

    const size_t chunk_size = 8;
    const size_t chunks = n / chunk_size;

#pragma omp parallel for num_threads(8)
    for(size_t i = 0; i < n; ++i)
    {
        __m256 a_line, b_line, c_line, r_line;
        for(size_t k = 0; k < n; ++k)
        {
            float c = da[i][k];
            a_line = _mm256_set_ps(c, c, c, c, c, c, c, c);
            for(size_t j = 0; j < chunks; ++j)
            {
                float mc[32] __attribute__((aligned(32))); 
                b_line = _mm256_load_ps(&db[k][j * chunk_size]);
                c_line = _mm256_load_ps(&dc[i][j * chunk_size]);
                r_line = _mm256_mul_ps(a_line, b_line);  
                r_line = _mm256_add_ps(r_line, c_line);
                _mm256_store_ps(&dc[i][j * chunk_size], r_line);
            } 

            for(size_t j = chunk_size * chunks; j < n; ++j)
            {
                dc[i][j] += c * db[k][j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    const int size = 4096;
    // void(*multi)(const matrix_t &, const matrix_t &, matrix_t &) = (strcmp(argv[3], "1") == 0) ? multi1 : multi2;
    matrix_t A(size), B(size), C(size);
    for(int i = 0; i < size; ++i){
        int base = i * size;
        for(int j = 0; j <size; ++j)
        {
            A.data[i][j] = (float)rand()/(float)(RAND_MAX/20);
            B.data[i][j] = (float)rand()/(float)(RAND_MAX/20);
        }
    }
    auto start = chrono::high_resolution_clock::now();
    multi2(A, B, C);
    auto end = chrono::high_resolution_clock::now();

    auto diff = end - start;
    printf("total time: %.3f ms\n", chrono::duration<float, milli>(diff).count());
    printf("%f\n", C.data[0][0]);
    return 0;
}
