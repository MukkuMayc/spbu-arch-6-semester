#include <mpi.h>
#include <iostream>

#include "ring.cpp"
#include "master_slave.cpp"
#include "everyone_to_everyone.cpp"

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    int group_rank, group_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &group_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    if (group_rank == 0) { std::cout << "\nRing communication\n\n"; }

    ring(MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (group_rank == 0) {
        std::cout << "\nMaster-slave communication\n\n";
        master(MPI_COMM_WORLD);
    }
    else {
        slave(MPI_COMM_WORLD, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (group_rank == 0) { std::cout << "\nEveryone to everyone communication\n\n"; }
    
    node(MPI_COMM_WORLD);

    // Finalize the MPI environment.
    MPI_Finalize();
}
