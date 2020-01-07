#include <SymEigsSolver.h> // Spectra's symmetric eigensolver

#include "CartanSyncProblem.h"
#include "SESync_utils.h"

#include "CartanSyncVariable.h"

namespace SESync {

void CartanSyncProblem::set_problem_data(const SparseMatrix& connection_Laplacian_T,
    size_t n_in, size_t m_in, size_t d_in,
    bool Cholesky, bool Preconditioning)
{
    // Initialize subclass data members
    n = n_in;
    m = m_in;
    d = d_in;

    // Matrices

    ConLapT = connection_Laplacian_T;

    use_Cholesky = Cholesky;
    use_Precon = Preconditioning;

    // Compute and cache the Cholesky factor L of ConLapT
    SparseMatrix redConLapT = ConLapT.topLeftCorner(ConLapT.rows()-1,ConLapT.cols()-1);
    L.compute( redConLapT );

    // General configuration
    SetUseGrad(true);
    SetUseHess(true);
}

void CartanSyncProblem::set_relaxation_rank(unsigned int rank)
{
    // Record this value
    r = rank;

    if (domain)
      delete domain;
    domain = new CartanSyncManifold(r,d,n);

    // Call the super function that sets the domain for this problem
    SetDomain(domain);

    // Allocate working space:
    // Concatenation of n [r x (d+1)] sub-matrices,
    // each formed by a [r x d] Stiefel and a [r x 1] Euclidean (linear)
    if (Y)
        delete Y;
    Y = new Matrix(r, n * (d+1));
}

Matrix CartanSyncProblem::Pi_product(const Matrix& X) const
{
    if (use_Cholesky)
        return X - SqrtOmega_AredT * L.solve(Ared_SqrtOmega * X);
    else {
        Eigen::MatrixXd PiX = X;
        for (unsigned int c = 0; c < X.cols(); c++) {
            // Eigen's SPQR support only supports solving with vectors(!) (i.e. 1-column matrices)
            PiX.col(c) = X.col(c) - SqrtOmega_AredT * QR->solve(X.col(c));
        }
        return PiX;
    }
}

Matrix CartanSyncProblem::Q_product(const Matrix& X) const
{
    return LGrho * X + TT_SqrtOmega * Pi_product(SqrtOmega_T * X);
}

Matrix CartanSyncProblem::compute_Lambda_blocks(const Matrix& Ystar) const
{
    Eigen::MatrixXd QYstarT = ConLapT * Y->transpose();
    // TODO: translation components should be zero for locally optimal solution

    Eigen::MatrixXd Lambda_blocks(d, n * d);

    // Preallocation of working space for computing the block elements of Lambda
    Eigen::MatrixXd B(d, d);

    // TODO: Careful with stride in block-vectors
    // TODO: Maybe define methods to access particular sections of memory
    //       e.g. R_i, t_i, T_i, etc.
    for (unsigned int i = 0; i < n; i++) {
        B = QYstarT.block(i * (d+1), 0, d, Ystar.rows()) * Ystar.block(0, i * (d+1), Ystar.rows(), d);
        Lambda_blocks.block(0, i * d, d, d) = .5 * (B + B.transpose());
    }
    return Lambda_blocks;
}

bool CartanSyncProblem::compute_Q_minus_Lambda_min_eig(
    const Matrix& Ystar, double& min_eigenvalue,
    Eigen::VectorXd& min_eigenvector, int max_iterations, double precision,
    unsigned int num_Lanczos_vectors) const
{
    // Set some convenient aliases for more compact code
    const Spectra::SELECT_EIGENVALUE LARGEST_MAGN = Spectra::SELECT_EIGENVALUE::LARGEST_MAGN;

    // First, compute the largest-magnitude eigenvalue of this matrix
    QMinusLambdaProdFunctor lm_op(this, Ystar);
    Spectra::SymEigsSolver<double, LARGEST_MAGN, QMinusLambdaProdFunctor>
        largest_magnitude_eigensolver(&lm_op, 1,std::min(num_Lanczos_vectors, n*(d+1)));
    largest_magnitude_eigensolver.init();

    int num_converged = largest_magnitude_eigensolver.compute(max_iterations, precision, LARGEST_MAGN);

    // Check convergence and bail out if necessary
    if (num_converged != 1)
        return false;

    double lambda_lm = largest_magnitude_eigensolver.eigenvalues()(0);

    if (lambda_lm < 0) {
        // The largest-magnitude eigenvalue is negative, and therefore also
        // the minimum eigenvalue, so just return this solution
        min_eigenvalue = lambda_lm;
        min_eigenvector = largest_magnitude_eigensolver.eigenvectors(1);
        min_eigenvector.normalize(); // Ensure that this is a unit vector
        return true;
    }

    /* The largest-magnitude eigenvalue is positive, and is therefore the
     * maximum eigenvalue. Therefore, after shifting the spectrum of Q - Lambda
     * by - 2*lambda_lm (by forming Q - Lambda - 2*lambda_max*I), the shifted
     * spectrum will line in the interval [lambda_min(A) - 2*  lambda_max(A),
     * -lambda_max*A]; in particular, the largest-magnitude eigenvalue of
     * Q - Lambda - 2*lambda_max*I is lambda_min - 2*lambda_max, with
     * corresponding eigenvector v_min; furthermore, the condition number sigma
     * of Q - Lambda - 2*lambda_max is then upper-bounded by 2 :-).
    */

    // TODO: Just set _sigma member in the previous QMinusLambdaProdFunctor
    QMinusLambdaProdFunctor min_shifted_op(this, Ystar, -2 * lambda_lm);

    Spectra::SymEigsSolver<double, LARGEST_MAGN, QMinusLambdaProdFunctor>
        min_eigensolver(&min_shifted_op, 1, std::min(num_Lanczos_vectors, n*(d+1)));

    // If Ystar is a critical point of F, then Ystar^T is also in the null space
    // of Q - Lambda(Ystar) (cf. Lemma 6 of the tech report), and therefore its
    // rows are eigenvectors corresponding to the eigenvalue 0. In the case that
    // the relaxation is exact, this is the *minimum* eigenvalue, and therefore
    // the rows of Ystar are exactly the eigenvectors that we're looking for.
    // On the other hand, if the relaxation is *not* exact, then
    // Q - Lambda(Ystar) has at least one strictly negative eigenvalue, and the
    // rows of Ystar are *unstable fixed points* for the Lanczos iterations.
    // Thus, we will take a slightly "fuzzed" version of the first row of Ystar
    // as an initialization for the Lanczos iterations; this allows for rapid
    // convergence in the case that the relaxation is exact (since are starting
    // close to a solution), while simultaneously allowing the iterations to
    // escape from this fixed point in the case that the relaxation is not exact.
    Eigen::VectorXd v0 = Ystar.row(0).transpose();
    Eigen::VectorXd perturbation(v0.size());
    perturbation.setRandom();
    perturbation.normalize();
    Eigen::VectorXd xinit = v0 + (.03 * v0.norm()) * perturbation; // Perturb v0 by ~3%

    // Use this to initialize the eigensolver
    min_eigensolver.init(xinit.data());
    num_converged = min_eigensolver.compute(max_iterations, precision, LARGEST_MAGN);

    if (num_converged != 1)
        return false;

    min_eigenvector = min_eigensolver.eigenvectors(1);
    min_eigenvector.normalize(); // Ensure that this is a unit vector
    min_eigenvalue = min_eigensolver.eigenvalues()(0) + 2 * lambda_lm;
    return true;
}

double CartanSyncProblem::f(ROPTLIB::Variable* x) const
{
    ROPTLIB::ProductElement* X = static_cast<ROPTLIB::ProductElement*>(x);
    CartanProd2Mat(*X, *Y);
    return 0.5*((*Y) * ConLapT * Y->transpose()).trace();
}

void CartanSyncProblem::EucGrad(ROPTLIB::Variable* x, ROPTLIB::Vector* g) const
{
    ROPTLIB::ProductElement* X = static_cast<ROPTLIB::ProductElement*>(x);
    ROPTLIB::ProductElement* G = static_cast<ROPTLIB::ProductElement*>(g);
    CartanProd2Mat(*X, *Y);
    Mat2CartanProd((*Y) * ConLapT, *G);
}

void CartanSyncProblem::EucHessianEta(ROPTLIB::Variable* x, ROPTLIB::Vector* v,
    ROPTLIB::Vector* Hv) const
{
    ROPTLIB::ProductElement* X = static_cast<ROPTLIB::ProductElement*>(x);
    ROPTLIB::ProductElement* V = static_cast<ROPTLIB::ProductElement*>(v);
    ROPTLIB::ProductElement* HV = static_cast<ROPTLIB::ProductElement*>(Hv);
    CartanProd2Mat(*V, *Y);
    Mat2CartanProd((*Y) * ConLapT, *HV);
}

void CartanSyncProblem::PreConditioner(ROPTLIB::Element *x,
    ROPTLIB::Element *inVec, ROPTLIB::Element *outVec) const
{
    if (use_Precon)
    {
        // Implementation of the naive preconditioning approach
        // proposed in our RAL paper for Cartan-Sync

        ROPTLIB::ProductElement* xi = static_cast<ROPTLIB::ProductElement*>(inVec);
        ROPTLIB::ProductElement* eta = static_cast<ROPTLIB::ProductElement*>(outVec);

        Matrix xiMat, etaMat;
        CartanProd2Mat(*xi, xiMat);

        etaMat = Eigen::MatrixXd::Zero(r,n*(d+1));

        // Solve the linear system ConLapT * linEta = linXi
        // Note this involves the Euclidean Hessian,
        // and this particular expression corresponds to the halved objective
        // where ConLapT is a PSD matrix (one zero eigenvalue)
        etaMat.leftCols(n*(d+1)-1) =
           (L.solve((xiMat.leftCols(n*(d+1)-1)).transpose())).transpose();

        Mat2CartanProd(etaMat, *eta);

        // Project preconditioned vector to tangent space again
        // outVec has been modified in the previous step through xi
        domain->Projection(x,outVec,outVec);
    }
    else
    {
        // no preconditioner.
        inVec->CopyTo(outVec);
    }
}
}
