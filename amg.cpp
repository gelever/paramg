/*! @file

    @brief A quick look into finite elements
*/

#include <stdio.h>
#include <assert.h>

#include "utilities.hpp"
#include "partition.hpp"
#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"

using namespace linalgcpp;
using namespace parlinalgcpp;

int main(int argc, char** argv)
{
    int myid;
    int num_procs;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // Partition Options
    const bool contig = true;
    const double ubal = 1.0;

    // Load Parallel Matrix
    ParMatrix A_par;
    {
        SparseMatrix<double> A_global = ReadBinaryMat("mat.bin");

        // Partition global A by processor
        auto proc_part = Partition(A_global, num_procs, contig, ubal);

        A_par = ParSplit(comm, A_global, proc_part);
    }
    const auto global_vertices = A_par.GlobalRows();
    const auto& vertex_starts = A_par.GetColStarts();

    // Partition local A into aggregates
    const auto& A_local = A_par.GetDiag();
    const int coarse_factor = 40;
    auto num_parts = std::max(1lu, (A_local.Rows() / coarse_factor));
    auto part = Partition(A_local, num_parts, contig, ubal);

    // Create Aggregate to Vertex Relationship
    auto agg_vertex_local = MakeAggVertex<double>(part);
    auto agg_starts = GenerateOffsets(comm, agg_vertex_local.Rows());

    auto num_agg_global = agg_starts.back();

    ParMatrix agg_v_global(comm, num_agg_global, global_vertices,
                           agg_starts, vertex_starts, agg_vertex_local);
    agg_v_global = 1.0;

    // Create Extended Aggregate to Vertex Relationship
    ParMatrix A_pattern = A_par;
    A_pattern = 1.0;

    ParMatrix agg_v_ext_global = agg_v_global.Mult(A_pattern);
    agg_v_ext_global = 1.0;

    // Find Approximate Lower Spectrum Eigenvectors
    const int num_evects = 2;
    std::vector<ParVector> evects(num_evects, {comm, global_vertices, vertex_starts});
    {
        // Setup Initial Guess
        evects[0] = 1.0 / std::sqrt(global_vertices);
        evects[1].Randomize();
        SubAvg(evects[1]);

        // Use CG as preconditioner
        const int num_iter = 50;
        ParCG cg(A_par, num_iter);

        LOBPCG(A_par, evects, &cg);
    }

    // Build Interpolation Matrix
    ParMatrix P;
    {
        int num_agg_local = agg_vertex_local.Rows();
        int num_vert_local = A_local.Rows();

        CooMatrix<double> P_local(num_vert_local, num_agg_local * num_evects);

        // Add local P by aggregated
        for (int i = 0; i < num_agg_local; ++i)
        {
            auto indices = agg_vertex_local.GetIndices(i);
            int size = indices.size();

            std::vector<double> norms(num_evects, 0.0);

            // Find local norm first
            for (int j = 0; j < num_evects; ++j)
            {
                for (auto& k : indices)
                {
                    norms[j] += evects[j][k] * evects[j][k];
                }

                norms[j] = std::sqrt(norms[j]);

                assert(norms[j] != 0.0);

                norms[j] = 1.0 / norms[j];
            }

            // Assemble normalized local P
            for (int k = 0; k < size; ++k)
            {
                auto row = indices[k];

                for (int j = 0; j < num_evects; ++j)
                {
                    auto col = (2 * i) + j;
                    auto val = evects[j][row];

                    P_local.Add(row, col, val * norms[j]);
                }
            }
        }

        // Create Global P from local P blocks
        auto P_block = P_local.ToSparse();
        auto p_starts = GenerateOffsets(comm, P_block.Cols());

        P = ParMatrix(comm, global_vertices, num_agg_global * num_evects, vertex_starts, p_starts, P_block);
    }

    auto A_coarse = RAP(A_par, P);


    if (myid == 2)
    {
        std::cout << "A coarse size: " << A_coarse.GetRowStarts();
        std::cout << "A coarse size: " << A_coarse.GetColStarts();
    }

    MPI_Finalize();
}
