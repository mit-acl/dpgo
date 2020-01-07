/** This class encapsulates an instance of the special Euclidean synchronization
* problem
*
* dmrosen 18 May 2017
* jbriales 24 Aug 2017
*/

#ifndef _CARTANSYNCPROBLEM_H_
#define _CARTANSYNCPROBLEM_H_

/** Use external matrix factorizations/linear solves provided by SuiteSparse
 * (SPQR and Cholmod) */

#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/SPQRSupport>
#include <Eigen/Sparse>

#include "RelativePoseMeasurement.h"
#include "SESync_types.h"
#include "SESync_utils.h"

#include "Problem.h"
#include "CartanSyncManifold.h"

#include <SESyncProblem.h>

namespace SESync {

/** The type of the sparse Cholesky factorization to use when applying
 * the preconditioner */
typedef Eigen::CholmodDecomposition<SparseMatrix> SparseCholeskyFactorization;

/** The type of the QR decomposition to use when applying the preconditioner */
typedef Eigen::SPQR<SparseMatrix> SparseQRFactorization;

class CartanSyncProblem : public SESyncProblem {

  /// Data-related matrices employed in Cartan-Sync approach (original SE-Sync)
  /// --------------------------------------------------------------------------

  /** The SE(d) Connection Laplacian gathering all the information */
  SparseMatrix ConLapT;

  /** The rotational connection Laplacian for the special Euclidean
*synchronization problem, cf. eq. 14 of the tech report */
  SparseMatrix LGrho;

  /** The weighted translational data matrix Omega^(1/2) T cf. e.g. eqs. 22-24
*of the tech report */
  SparseMatrix SqrtOmega_T;

  /** The transpose of the above matrix; we cache this for computational
 * efficiency, since it's used frequently */
  SparseMatrix TT_SqrtOmega;

  /** The weighted reduced oriented incidence matrix Ared Omega^(1/2), cf. e.g.
*eq. 39 of the tech report */
  SparseMatrix Ared_SqrtOmega;

  /** The transpose of the above matrix; we cache this for computational
 * efficiency, since it's used frequently */
  SparseMatrix SqrtOmega_AredT;

  /** An Eigen sparse linear solver that encodes the Cholesky factor L used
*in the solution of the linear problem withing the preconditioner. */
  SparseCholeskyFactorization L;

  /** An Eigen sparse linear solver that encodes the QR factorization used
*in the solution of the linear problem withing the preconditioner. */
  // When using Eigen::SPQR, the destructor causes a segfault if this variable
  // isn't explicitly initialized (i.e. not just default-constructed)
  SparseQRFactorization* QR = nullptr;

  /** A Boolean variable determining whether to use the Cholesky or QR
 * decompositions for computing the orthogonal projection */
  bool use_Cholesky;

  /** A Boolean variable determining whether to use Preconditioning within
 * the RTR approach */
  bool use_Precon;

  /** The product manifold of lifted SE(d)'s that is the domain of our method */
  CartanSyncManifold* domain = nullptr;

  /** Preallocated working space for ProductElement <=> Matrix conversion */
  Matrix* Y = nullptr;

 public:
  /** Default constructor; doesn't actually do anything */
  CartanSyncProblem() {}

  /** Constructor using a vector of relative pose measurements */
  CartanSyncProblem(
      const std::vector<SESync::RelativePoseMeasurement>& measurements) {
    SparseMatrix ConLapT = construct_connection_Laplacian_T(measurements);
    size_t n,m,d;
    d = (!measurements.empty() ? measurements[0].t.size() : 0);
    m = measurements.size();
    n = ConLapT.rows()/(d+1);

    set_problem_data(ConLapT,
                     n,m,d);
  }

  /** This function constructs the special Euclidean synchronization problem
*from the passed data matrices */
  CartanSyncProblem(const SparseMatrix& connection_Laplacian_T,
                    size_t n, size_t m, size_t d,
                    bool Cholesky = true,
                    bool Preconditioning = true) {
    // This is just a passthrough to the initialization function
    set_problem_data(connection_Laplacian_T,
                     n,m,d,
                     Cholesky,Preconditioning);
  }

  /** This function initializes the special Euclidean synchronization problem
*using the passed data matrices */
  void set_problem_data(const SparseMatrix& connection_Laplacian_T,
                        size_t n, size_t m, size_t d,
                        bool Cholesky = true,
                        bool Preconditioning = true);

  /** Given a matrix X, this function computes and returns the orthogonal
*projection Pi * X */
  Matrix Pi_product(const Matrix& X) const;

  /** This function computes and returns the product QX */
  Matrix Q_product(const Matrix& X) const;

  /** Given Y* in St(r,d)^n, this function computes and returns a d x nd matrix
*comprised of the dxd block elements of the associated block-diagonal
*Lagrange multiplier matrix Lambda(Y) */
  Matrix compute_Lambda_blocks(const Matrix& Ystar) const;

  /** Given a critical point Y* in the domain of the optimization problem, this
*function computes the smallest eigenvalue lambda_min of Q - Lambda and its
*associated eigenvector v.  Returns a Boolean value indicating whether the
*Lanczos method used to estimate the smallest eigenpair convergence to
*within the required tolerance. */
  bool compute_Q_minus_Lambda_min_eig(
      const Matrix& Ystar, double& min_eigenvalue,
      Eigen::VectorXd& min_eigenvector, int max_iterations = 10000,
      double precision = 1e-6, unsigned int num_Lanczos_vectors = 20) const;

  /** Set the (maximum) partial lifting rank for Cartan-Sync */
  void set_relaxation_rank(unsigned int rank);

  /// ACCESSORS

  /** Get a const pointer to the Stiefel product manifold */
  const ROPTLIB::ProductManifold* get_Stiefel_product_manifold() const {
    return domain;
  }

  /// OVERRIDDEN PURE VIRTUAL BASE CLASS (ROPTLIB::PROBLEM) FUNCTIONS

  /** Evaluates the problem objective */
  double f(ROPTLIB::Variable* x) const;

  /** Evaluates the Euclidean gradient of the function */
  void EucGrad(ROPTLIB::Variable* x, ROPTLIB::Vector* g) const;

  /** Evaluates the action of the Euclidean Hessian of the function */
  void EucHessianEta(ROPTLIB::Variable* x, ROPTLIB::Vector* v,
                     ROPTLIB::Vector* Hv) const;

  /** Evaluates the action of the Preconditioner for the Hessian of the function */
  void PreConditioner(ROPTLIB::Variable* x, ROPTLIB::Vector* inVec,
                      ROPTLIB::Vector* outVec) const;

  ~CartanSyncProblem() {
//    if (Stdr) delete Stdr;
    if (domain) delete domain;
    if (Y) delete Y;

    if (QR) delete QR;
  }

  /** This is a lightweight struct used in conjunction with Spectra to compute
*the minimum eigenvalue and eigenvector of Q - Lambda; it has a single
*nontrivial function, perform_op(x,y), that computes and returns the product
*y = (Q - Lambda + sigma*I) x */
  struct QMinusLambdaProdFunctor {
    const CartanSyncProblem* _problem;
    Matrix _Lambda_blocks;
    int _rows;
    int _cols;
    int _dim;
    double _sigma;

    QMinusLambdaProdFunctor(const CartanSyncProblem* prob, const Matrix& Ystar,
                            double sigma = 0)
        : _problem(prob),
          _rows((prob->dimension()+1) * prob->num_poses()),
          _cols((prob->dimension()+1) * prob->num_poses()),
          _dim(prob->dimension()),
          _sigma(sigma) {
      // Compute and cache this on construction
      _Lambda_blocks = prob->compute_Lambda_blocks(Ystar);
    }
    int rows() const { return _rows; }
    int cols() const { return _cols; }

    void perform_op(double* x, double* y) const {
      Eigen::Map<Eigen::VectorXd> X(x, _cols);
      Eigen::Map<Eigen::VectorXd> Y(y, _rows);

      /* QMinusLambda has the form (Q - Lambda) where
       *  Q is a sparse matrix,
       *  Lambda is block-diagonal,
       * so for efficiency and to avoid having to copy the Q matrix,
       * the vector-product (Q-Lambda)*x is decomposed into two terms:
       *  Q*x - Lambda*x
      */

      Y = _problem->ConLapT * X;

      for (unsigned int i = 0; i < _problem->num_poses(); i++)
        Y.segment(i * (_dim+1), _dim) -=
            _Lambda_blocks.block(0, i * _dim, _dim, _dim) *
            X.segment(i * (_dim+1), _dim);

      if (_sigma != 0) Y += _sigma * X;
    }
  };

  /** Hacky workaround: store the gradient norm tolerance for the RTR
   * optimization here, so that we can access it through a plain C-style
   * callback that passes an ROPTLIB::Problem* instance as an argument */

  double RTR_gradient_norm_tolerance;
};
}
#endif  // _CARTANSYNCPROBLEM_H_
