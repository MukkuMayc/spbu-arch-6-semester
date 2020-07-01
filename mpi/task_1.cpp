#include <mpi.h>
#include <stdio.h>

#include "ring.cpp"

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    ring(MPI_COMM_WORLD);
    
    // Finalize the MPI environment.
    MPI_Finalize();
}
