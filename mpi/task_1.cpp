#include <mpi.h>
#include <iostream>

#include "ring.cpp"
#include "master_slave.cpp"

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    int group_rank, group_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &group_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    if (group_rank == 0) { std::cout << "Ring communication\n"; }

    ring(MPI_COMM_WORLD);

    if (group_rank == 0) {
        std::cout << "Master-slave communication\n";
        master(MPI_COMM_WORLD);
    }
    else {
        slave(MPI_COMM_WORLD, 0);
    }


    // Finalize the MPI environment.
    MPI_Finalize();
}
