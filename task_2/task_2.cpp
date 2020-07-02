#include <vector>
#include "mergesort.cpp"
#include <cassert>
#include <chrono>
#include <iomanip>

template<typename Functor>
void print_exec_time(Functor f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                            (end - start).count() / 1e9;
    std::cout << "Elapsed time: " << duration << std::endl;
}

template<typename Functor>
double exec_time(Functor f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                            (end - start).count() / 1e9;
    return duration;
}

int main() {
    MPI_Init(NULL, NULL);

    int group_rank, group_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &group_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    int ssize = 5;
    double time_normal[ssize];
    double time_parallel[ssize];
    int experiments_size = 10;
    for (int i = 0; i < ssize; time_normal[i] = 0., time_parallel[i++] = 0.);
    // варьируем размер сортируемого массива от 1e6 до 5e6
    for (uint k = 0; k < ssize; ++k) {
        uint size = 1e5 + k * 1e5; 
        // совершаем 10 итераций для каждого размера
        for (uint j = 0; j < experiments_size; ++j) {
            if (group_rank == 0) {std::cout << "SIZE: " << size << "\n\n";}
            // создаём два массива, один для параллельной сортировки, другой для обычной
            int* arr1;
            int* arr2;
            if (group_rank == 0) {
                arr1 = generate_array(size);
                arr2 = new int[size];
                std::copy(arr1, arr1 + size, arr2);
            }

            MPI_Barrier(MPI_COMM_WORLD);

            // если мы главный процесс, то сохраняем время выполнения параллельной сортировки сортировки
            if (group_rank == 0) {
                std::cout << "Parallel\n";
                time_parallel[k] += exec_time(std::bind(mergesort_parallel, arr1, size));
                std::cout << std::endl;
            }
            else { mergesort_parallel(arr1, size); }

            MPI_Barrier(MPI_COMM_WORLD);

            // если главный процесс, то выполняем обычную сортировку, проверяем отсортированность
            if (group_rank == 0) {
                std::cout << "Normal\n";
                time_normal[k] += exec_time(std::bind(mergesort<int*>, arr2, arr2 + size));
                assert(is_sorted(arr1, size));
                assert(is_equal(arr1, arr2, size));

                if (size <= 50) {
                    for (int i = 0; i < size; ++i) {
                        std::cout << ' ' << arr1[i];
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    // напечатаем среднее время выполнения
    if (group_rank == 0) {
        std::cout << "TIME" << std::endl;
        std::cout << "SIZE\t\t";
        for (int i = 0; i < ssize; ++i) {
            std::cout << 1e5 + i * 1e5 << "\t\t";
        }
        std::cout << std::endl;

        std::cout << "NORMAL\t\t";
        for (int i = 0; i < ssize; ++i) {
            std::cout << std::setprecision(3) << time_normal[i] / experiments_size << "\t\t";
        }
        std::cout << std::endl;

        std::cout << "PARALLEL\t";
        for (int i = 0; i < ssize; ++i) {
            std::cout << std::setprecision(3) << time_parallel[i] / experiments_size << "\t\t";
        }
        std::cout << std::endl;
    }
    
    MPI_Finalize();
}