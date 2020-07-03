#include "stdio.h"
#include <assert.h>

#define N   10
#define M   5

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void add_normal(int *a, int *b, int *c) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            int ind = i * M + j;
            c[ind] = a[ind] + b[ind];
        }
    }
}

__global__ void add_parallel(int *a, int *b, int *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < M) {
        int ind = i * M + j;
        c[ind] = a[ind] + b[ind];
    }
}

bool is_equal(int* a, int* b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int ind = i * M + j;
            if (a[ind] != b[ind]) { return false; }
        }
    }
    return true;
}

int main( void ) {
    int a[N * M], b[N * M], c[2][N * M];

    for (int i = 0; i < N * M; i++) {
        a[i] = -i;
        b[i] = i * i;
    }
    int *da, *db, *dc;
    int byte_size = N * M * sizeof(int);
    gpuErrchk( cudaMalloc((void **)&da, byte_size) );
    gpuErrchk( cudaMalloc((void **)&db, byte_size) );
    gpuErrchk( cudaMalloc((void **)&dc, byte_size) );

    gpuErrchk( cudaMemcpy(da, a, byte_size, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(db, b, byte_size, cudaMemcpyHostToDevice));

    // вычисляем результат каждой функции и записываем в соответствующий c[i]
    add_normal(a, b, c[0]);

    dim3 block(8, 4);
    dim3 grid(ceil(N / ((float)block.x)), ceil(M / ((float)block.y)));

    printf("%d/%d, %d/%d\n", grid.x, grid.y, block.x, block.y);
    
    add_parallel<<<grid, block>>>(da, db, dc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpy(c[1], dc, byte_size, cudaMemcpyDeviceToHost) );


    // display the results
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                printf("%d\t", c[k][i * M + j]);    
            }
            printf("\n");
        }
    }

    // проверяем, чтобы результаты совпадали
    for (int i = 1; i < 2; ++i) {
        assert(is_equal(c[i - 1], c[i], N, M));
    }

    printf("ASSERT COMPLETED");

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}
