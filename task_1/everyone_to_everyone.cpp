#include <mpi.h>
#include <iostream>
#include <string>
#include <random>

void node(MPI_Comm comm) {
    int group_rank, group_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &group_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    std::default_random_engine gen((std::random_device())());
    std::uniform_int_distribution<int> dist(100, 999);

    int buf[group_size];
    buf[group_rank] = dist(gen);
    // for (int i = 0; i < group_size; buf[i++] = i);

    for (int i = 0; i < group_size; ++i) {
        MPI_Bcast(buf + i, 1, MPI_INT, i, comm);
    }

    for (int i = 0; i < group_size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i != group_rank) { continue; }

        std::cout << "Process " << group_rank << ", my array:";
        for (int i = 0; i < group_size; std::cout << ' ' << buf[i++]);
        std::cout << std::endl;
    }
}
