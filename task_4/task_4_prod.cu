#include "stdio.h"
#include <assert.h>

#define N   16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void prod_normal(const int* a, const int* b, int *c) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int ind = i * N + j;
            c[ind] = 0;
            for (int k = 0; k < N; ++k) {
                c[ind] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

__global__ void prod_parallel(int* a, int* b, int* c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        int val = 0;
        for (int k = 0; k < N; ++k) {
            val += a[x * N + k] * b[k * N + y];
        }
        c[x * N + y] = val;
    }
}

#define BLOCK_SIZE 4

__global__ void prod_parallel_shared(int* A, int* B, int* C) {
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;
    int row = threadIdx.x;
    int col = threadIdx.y;

    int val = 0;

    int rowC = block_row * BLOCK_SIZE + row;
    int colC = block_col * BLOCK_SIZE + col;

    for (int m = 0; m < (N / BLOCK_SIZE); ++m) {
        __shared__ int sA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ int sB[BLOCK_SIZE][BLOCK_SIZE];

        int indA =  rowC * N + (m * BLOCK_SIZE + col);
        int indB =  (m * BLOCK_SIZE + row) * N + colC;
        sA[row][col] = A[indA];
        sB[row][col] = B[indB];
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            val += sA[row][i] * sB[i][col];
        }

        __syncthreads();
    }

    C[rowC * N + colC] = val;
}

bool is_equal(const int* a, const int* b, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            int ind = i * m + j;
            if (a[ind] != b[ind]) { return false; }
        }
    }
    return true;
}

void print_matrix(const int* a, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            printf("%d\t", a[i * m + j]);
        }
        printf("\n");
    }
}

int main( void ) {
    int a[N * N], b[N * N], c[3][N * N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i * N + j] = i * N + j;
            b[i * N + j] = i == j ? 1 : 0;
        }
    }

    int *da, *db, *dc;
    int byte_size = N * N * sizeof(int);
    gpuErrchk( cudaMalloc((void **)&da, byte_size) );
    gpuErrchk( cudaMalloc((void **)&db, byte_size) );
    gpuErrchk( cudaMalloc((void **)&dc, byte_size) );

    gpuErrchk( cudaMemcpy(da, a, byte_size, cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(db, b, byte_size, cudaMemcpyHostToDevice));

    // вычисляем результат каждой функции и записываем в соответствующий c[i]
    prod_normal(a, b, c[0]);

    dim3 block(4, 4);
    dim3 grid(ceil(N / ((float)block.x)), ceil(N / ((float)block.y)));

    printf("%d/%d, %d/%d\n", grid.x, grid.y, block.x, block.y);
    
    prod_parallel<<<grid, block>>>(da, db, dc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpy(c[1], dc, byte_size, cudaMemcpyDeviceToHost) );

    prod_parallel_shared<<<grid, block>>>(da, db, dc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaMemcpy(c[2], dc, byte_size, cudaMemcpyDeviceToHost) );


    if (N <= 16) {
        // display the results
        printf("A:\n");
        print_matrix(a, N, N);
        printf("B:\n");
        print_matrix(b, N, N);
        printf("C1:\n");
        print_matrix(c[0], N, N);
        printf("C2:\n");
        print_matrix(c[1], N, N);
        printf("C3:\n");
        print_matrix(c[2], N, N);

    }

    // проверяем, чтобы результаты совпадали
    for (int i = 1; i < 3; ++i) {
        assert(is_equal(c[i - 1], c[i], N, N));
    }

    printf("ASSERT COMPLETED");

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}
