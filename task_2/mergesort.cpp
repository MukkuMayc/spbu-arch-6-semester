#include <mpi.h>
#include <iostream>
#include <string>
#include <random>
#include <iterator>
#include <algorithm> // for std::inplace_merge
#include <functional> // for std::less
#include <limits>


bool is_sorted(const int* a, int size) {
    for (int i = 1; i < size; ++i) {
        if (a[i - 1] > a[i]) { return false; };
    }
    return true;
}

bool is_equal(const int* a, const int* b, int size) {
    for (int i = 0; i < size; ++i) {
        if (a[i] != b[i]) { return false; };
    }
    return true;
}

int* generate_array(uint size) {
    std::default_random_engine gen((std::random_device())());
    std::uniform_int_distribution<int> dist(100, 999);

    int* arr = new int[size];
    for (int i = 0; i < size; arr[i++] = dist(gen));
    
    return arr;
}

//source: https://rosettacode.org/wiki/Sorting_algorithms/Merge_sort#C.2B.2B
template<typename RandomAccessIterator, typename Order>
void mergesort(RandomAccessIterator first, RandomAccessIterator last, Order order) {
    if (last - first > 1) {
        RandomAccessIterator middle = first + (last - first) / 2;
        mergesort(first, middle, order);
        mergesort(middle, last, order);
        std::inplace_merge(first, middle, last, order);
    }
}

template<typename RandomAccessIterator>
void mergesort(RandomAccessIterator first, RandomAccessIterator last) {
    mergesort(first, last, std::less<typename std::iterator_traits<RandomAccessIterator>::value_type>());
}

void mergesort_parallel(int* arr, int size) {
    int group_rank, group_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &group_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    int new_size = size - size % group_size + group_size * (size % group_size);
    int* new_arr;

    if (group_rank == 0) {
        new_arr = new int[new_size];
        std::copy(arr, arr + size, new_arr);
        for (int i = size; i < new_size; new_arr[i++] = std::numeric_limits<int>::max());
    }

    int recv_size = new_size / group_size;
    int recv_buf[recv_size];

    MPI_Scatter(new_arr, recv_size, MPI_INT, recv_buf, recv_size, MPI_INT, 0, MPI_COMM_WORLD);

    mergesort(recv_buf, recv_buf + recv_size);

    MPI_Gather(recv_buf, recv_size, MPI_INT, new_arr, recv_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (group_rank == 0) {
        for (int i = 2; i < group_size + 1; ++i) {
            std::inplace_merge(new_arr, new_arr + recv_size * (i - 1), new_arr + recv_size * i);
        }
        std::copy(new_arr, new_arr + size, arr);
    }
}