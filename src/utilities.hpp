/*! @file */

#ifndef UTILITIES_HPP__
#define UTILITIES_HPP__

#include "linalgcpp.hpp"

void WriteSol(const linalgcpp::Vector<double>& sol, const std::string& file_name = "sol.gf");

template <typename T = int>
linalgcpp::SparseMatrix<T> MakeEdgeVertex(const linalgcpp::SparseMatrix<T>& mat)
{
    assert(mat.Rows() == mat.Cols());
    assert((mat.nnz() - mat.Rows()) % 2 == 0);

    const size_t num_vertices = mat.Rows();
    const size_t num_edges = (mat.nnz() - mat.Rows()) / 2;

    std::vector<int> indptr(num_edges + 1);
    std::vector<int> indices(num_edges * 2);
    std::vector<T> data(num_edges * 2, 1);

    indptr[0] = 0;

    int count = 0;

    for (size_t i = 0; i < num_vertices; ++i)
    {
        for (int j = mat.GetIndptr()[i]; j < mat.GetIndptr()[i + 1]; ++j)
        {
            const size_t col = mat.GetIndices()[j];

            if (i < col)
            {
                indices[count++] = i;
                indices[count++] = col;

                indptr[count / 2] = count;
            }
        }
    }

    return linalgcpp::SparseMatrix<T>(indptr, indices, data, num_edges, num_vertices);
}

template <typename T = int>
linalgcpp::SparseMatrix<T> RestrictInterior(const linalgcpp::SparseMatrix<T>& mat)
{
    std::vector<int> indptr(mat.Rows() + 1);
    std::vector<int> indices;

    indptr[0] = 0;

    for (size_t i = 0; i < mat.Rows(); ++i)
    {
        for (int j = mat.GetIndptr()[i]; j < mat.GetIndptr()[i + 1]; ++j)
        {
            if (mat.GetData()[j] >= 2)
            {
                indices.push_back(mat.GetIndices()[j]);
            }
        }

        indptr[i + 1] = indices.size();
    }

    std::vector<T> data(indices.size(), 1);

    return linalgcpp::SparseMatrix<T>(indptr, indices, data, mat.Rows(), mat.Cols());
}

/*
template <typename T>
void EliminateRowCol(int index, linalgcpp::SparseMatrix<T>& mat, linalgcpp::Vector<double>& rhs)
{
    const auto& indptr = mat.GetIndptr();
    const auto& indices = mat.GetIndices();
    auto& data = mat.GetData();

    const int rows = mat.Rows();

    for (int i = 0; i < rows; ++i)
    {
        const int row = i;

        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            const int col = indices[j];

            if (row == index && col == index)
            {
                data[j] = 1.0;
            }
            else if (row == index || col == index)
            {
                data[j] = 0.0;
            }

        }

        if (row == index)
        {
            rhs[index] = 0.0;
        }
    }
}

template <typename T>
void EliminateRow(const std::vector<int>& rows, linalgcpp::SparseMatrix<T>& mat)
{
    const auto& indptr = mat.GetIndptr();
    const auto& indices = mat.GetIndices();
    auto& data = mat.GetData();

    for (const auto& row : rows)
    {
        for (int j = indptr[row]; j < indptr[row + 1]; ++j)
        {
            const int col = indices[j];

            if (row == col)
            {
                data[j] = 1.0;
            }
            else
            {
                data[j] = 0.0;
            }
        }
    }
}

template <typename T>
void EliminateRowCol(const std::vector<int>& indices, linalgcpp::SparseMatrix<T>& mat)
{
    EliminateRow(indices, mat);
    mat = mat.Transpose();
    EliminateRow(indices, mat);
    mat = mat.Transpose();
}

void EliminateRhs(const std::vector<int>& indices, linalgcpp::Vector<double>& rhs)
{
    for (const auto& i : indices)
    {
        rhs[i] = 0.0;
    }
}
*/

template <typename T=int>
linalgcpp::SparseMatrix<T> MakeAggVertex(std::vector<int> partition)
{
    const int num_parts = *std::max_element(std::begin(partition), std::end(partition)) + 1;
    const int num_vert = partition.size();

    std::vector<int> indptr(num_vert + 1);
    std::vector<T> data(num_vert, 1);

    std::iota(std::begin(indptr), std::end(indptr), 0);

    linalgcpp::SparseMatrix<T> vertex_agg(std::move(indptr), std::move(partition), std::move(data), num_vert, num_parts);

    return vertex_agg.Transpose();
}

linalgcpp::DenseMatrix OrthoBack(linalgcpp::DenseMatrix& A, size_t max_evects);
linalgcpp::DenseMatrix OrthoFront(linalgcpp::DenseMatrix& A, size_t max_evects);

#endif // UTILITIES_HPP__
