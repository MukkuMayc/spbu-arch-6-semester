#include <mpi.h>
#include <string>
#include <iostream>

void print_group_status(int group_size, int group_rank) {
    printf("GROUP RANK/SIZE: %d/%d\n", group_rank, group_size);
}

void ring(MPI_Comm group_comm)  {
    int group_rank, group_size;
    MPI_Comm_rank(group_comm, &group_rank);
    MPI_Comm_size(group_comm, &group_size);

    if (group_rank != 0) {
        int buf_size = 32;
        char inpmsg[buf_size];
        for (int i = 0; i < buf_size; inpmsg[i++] = 0);
        MPI_Status stat;
        int proc_prev = group_rank - 1;
        MPI_Recv(inpmsg, buf_size - 1, MPI_CHAR, proc_prev, 1, group_comm, &stat);
        print_group_status(group_size, group_rank);
        std::cout << "Receive message from " << proc_prev << ". Message: " <<
            inpmsg << std::endl;
    }
    else {
        std::cout << "I'm first process!\n";
        print_group_status(group_size, group_rank);
    }

    if (group_rank != group_size - 1) {
        int dest = group_rank + 1;
        std::string outmsg = "You're next, process " + std::to_string(dest);
        MPI_Send(outmsg.data(), outmsg.length(), MPI_CHAR, dest, 1, group_comm);
    }
    else {
        std::cout << "I'm last process!\n";
    }
}