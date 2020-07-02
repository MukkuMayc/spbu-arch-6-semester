#include <mpi.h>
#include <iostream>
#include <vector>

void master(MPI_Comm comm) {
    int group_rank, group_size;
    MPI_Comm_rank(comm, &group_rank);
    MPI_Comm_size(comm, &group_size);
    int mes_size = 32;
    int buf_size = mes_size * group_size;
    char inpmsg[buf_size];
    for (int i = 0; i < buf_size; inpmsg[i++] = 0);

    std::vector<MPI_Request> reqs;
    reqs.reserve(group_size);

    for (int i = 0; i < group_size; ++i) {
        if (i == group_rank) { continue; }
        reqs.emplace_back();
        MPI_Irecv(inpmsg + i * mes_size, mes_size - 1, MPI_CHAR, i, 1, comm, &reqs[reqs.size() - 1]);
    }

    std::vector<MPI_Status> statuses;
    statuses.resize(reqs.size());
    MPI_Waitall(reqs.size(), reqs.data(), statuses.data());

    for (int i = 0; i < group_size; ++i) {
        if (i == group_rank) { continue; }

        std::cout << (inpmsg + i * mes_size) << std::endl;
    }

}

void slave(MPI_Comm comm, int master_rank) {
    int group_rank;
    MPI_Comm_rank(comm, &group_rank);
    std::string mes{"Greetings from slave " + std::to_string(group_rank)};
    MPI_Send(mes.data(), mes.size(), MPI_CHAR, master_rank, 1, comm);
}