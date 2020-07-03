#include "stdio.h"
#include <assert.h>

#define N   10

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void add_1_thread( int *a, int *b, int *c) {
    int tid = 0;    // this is CPU zero, so we start at zero
    while (tid < 10) {
        c[tid] = a[tid] + b[tid];
        tid += 1;   // we have one CPU, so we increment by one
    }
}

__global__ void add_1_per_block( int *a, int *b, int *c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

__global__ void add_1_per_thread( int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

__global__ void add_multiple_threads_and_blocks( int *a, int *b, int *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}


bool is_equal(int* a, int* b, int size) {
    for (int i = 0; i < size; ++i) {
        if (a[i] != b[i]) { return false; }
    }
    return true;
}

int main( void ) {
    // создаём по вектору c для каждой функции
    int a[N], b[N], c[4][N];

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }
    int *da, *db, *dc;
    int byte_size = N * sizeof(int);
    gpuErrchk( cudaMalloc((void **)&da, byte_size) );
    gpuErrchk( cudaMalloc((void **)&db, byte_size) );
    gpuErrchk( cudaMalloc((void **)&dc, byte_size) );

    gpuErrchk( cudaMemcpy(da, a, byte_size, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(db, b, byte_size, cudaMemcpyHostToDevice));

    // вычисляем результат каждой функции и записываем в соответствующий c[i]
    add_1_thread<<<1, 1>>>(da, db, dc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpy(c[0], dc, byte_size, cudaMemcpyDeviceToHost) );

    add_1_per_block<<<N, 1>>>(da, db, dc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpy(c[1], dc, byte_size, cudaMemcpyDeviceToHost) );

    add_1_per_thread<<<1, N>>>(da, db, dc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpy(c[2], dc, byte_size, cudaMemcpyDeviceToHost) );

    add_multiple_threads_and_blocks<<<20, 50>>>(da, db, dc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpy(c[3], dc, byte_size, cudaMemcpyDeviceToHost) );
    
    // проверяем, чтобы результаты совпадали
    for (int i = 1; i < 4; ++i) {
        assert(is_equal(c[i - 1], c[i], N));
    }

    printf("ASSERT COMPLETED");

    // display the results
    // for (int i = 0; i < N; i++) {
    //     printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    // }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}
