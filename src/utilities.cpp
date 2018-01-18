#include "utilities.hpp"


using namespace linalgcpp;

DenseMatrix OrthoFront(DenseMatrix& A, size_t max_evects)
{
    const int size = std::min(std::min(A.Cols(), max_evects), A.Rows());

    DenseMatrix ortho(A.Rows(), size);
    if (A(0, 0) < 0)
    {
        //A *= -1.0;
    }

    for (int i = 0; i < size; ++i)
    {
        DenseMatrix v = A.GetCol(i, i + 1);
        DenseMatrix v_T = v.Transpose();
        const double l2norm = std::sqrt(v_T.Mult(v)(0, 0));
        v /= l2norm;
        v_T /= l2norm;

        DenseMatrix v_TA = v_T.Mult(A);
        DenseMatrix v_v_TA= v.Mult(v_TA);

        A -= v_v_TA;

        ortho.SetCol(i, v);
    }

    A = ortho;
    return ortho;
}

DenseMatrix OrthoBack(DenseMatrix& A, size_t max_evects)
{
    const int size = std::min(std::min(A.Cols(), max_evects), A.Rows());

    DenseMatrix ortho2 = A.GetCol(A.Cols() - size, A.Cols());
    ortho2.QR();
    return ortho2;
}
