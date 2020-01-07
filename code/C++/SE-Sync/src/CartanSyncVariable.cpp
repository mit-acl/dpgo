
#include "CartanSyncVariable.h"

// TODO: Remove dependencies on Cartan2... methods here
#include "SESync_utils.h"

#include "SESync_types.h"

// For the ROPTLIB types stored internally
#include "Stiefel.h"
#include "Euclidean.h"

#include "Eigen/SVD" // required for SVD decomposition in unlift method

namespace SESync{

  CartanVariable::CartanVariable(integer r, integer d) :
    ProductElement(2,
                    new ROPTLIB::StieVariable(r, d),1,
                    new ROPTLIB::EucVariable(r),1)
  { }

  CartanSyncVariable::CartanSyncVariable(integer r, integer d, integer n) :
    ProductElement(1, new CartanVariable(r,d), n), r(r), d(d), n(n)
  {
    // Interface data in the ROPTLIB variable as an Eigen matrix
//    Y = Eigen::Map<Eigen::MatrixXd>(
//          (double *)this->ObtainReadData(), r, n*(d+1));
  }

  CartanSyncVariable::~CartanSyncVariable(void)
  {
  }

  CartanSyncVariable *CartanSyncVariable::ConstructEmpty() const
  {
    return new CartanSyncVariable(r,d,n);
  }

  void CartanSyncVariable::lift( CartanSyncVariable &Xlift) const
  {
    Matrix Y;
    CartanProd2Mat(*static_cast<const ROPTLIB::ProductElement*>(this),Y);

    assert( Xlift.r > this->r && "the lifted domain must be larger" );
    // Construct a new matrix Ylift by augmenting Y with a new row of zeros
    // at the bottom
    Matrix Ylift;
    Ylift = Eigen::MatrixXd::Zero(Xlift.r, n*(d+1));
    Ylift.topRows(r) = Y;
    Mat2CartanProd(Ylift,Xlift);

    // Compact version when .mat() is implemented?
//    Xlift.mat() = Eigen::MatrixXd::Zero(Xlift.r, n*(d+1));;
//    Xlift.mat().topRows(r) = this->mat();
  }

  void CartanSyncVariable::unlift( CartanSyncVariable &Xunlift) const
  {
    Matrix Y;
    // TODO: No resize needed?
    CartanProd2Mat(*static_cast<const ROPTLIB::ProductElement*>(this),Y);

    /* Clean the nullspace component due to translation observability */
    /* The solution may be contaminated by components coming from
    * the translation observability nullspace, so we remove those
    * by projecting onto the range space (orthogonal to nullspace)
    */
    // Build the nullspace vector
    Eigen::VectorXd temp = Eigen::VectorXd::Zero(d+1);
//    temp.tail(1) = 1.0;
    temp(d) = 1.0;
//    Eigen::VectorXd uNull = temp.replicate(n);
    Eigen::VectorXd uNull = temp.replicate(n,1);
    uNull.normalize(); //inplace normalization
    // Compute the projection of the point onto the nullspace
    Matrix YprojNull = (Y*uNull)*uNull.transpose();
    // Obtain a clean version (projected onto the range space) by substracting
    // the projection onto the nullspace
    // TODO: Do computation in place? No, not possible since it's const
    Matrix YprojRange = Y - YprojNull;

    /* Recover the low-rank component in the current point.
    * If the relaxation is tight, this approximation should be exact,
    * that is, the dropped singular values should be zero.
    */
    integer new_r = Xunlift.r;
    assert( new_r < r ); // unlift only if the new domain is smaller

    // TODO: This first part is low-rank approximation, and could be refactored DRY
    // First, compute a thin SVD of Y
    Eigen::JacobiSVD<Matrix> svd(YprojRange, Eigen::ComputeThinV);

    Eigen::VectorXd sigmas = svd.singularValues();
    // Construct a diagonal matrix comprised of the first new_r singular values
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> Xid(new_r);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic>::DiagonalVectorType &diagonal =
        Xid.diagonal();
    for (unsigned int i = 0; i < new_r; i++)
      diagonal(i) = sigmas(i);

    Matrix Yunlift = Xid * svd.matrixV().leftCols(new_r).transpose();
//    // TODO: Apply retraction on this?
//    // TODO: Fix possible orientation issues out of here, in common method?

    Mat2CartanProd(Yunlift,Xunlift);
  }

  Matrix CartanSyncVariable::mat()
  {
    // In order to avoid conflicts when writing the space variables,
    // make sure there is a single copy of this SmartSpace
    assert( *(this->GetSharedTimes()) == 1 && "Shared or non-existing space" );

    // Interface data in the ROPTLIB variable as an Eigen matrix
    Matrix Y = Eigen::Map<Matrix>(
          (double *)this->ObtainReadData(), r, n*(d+1));
    // Line below should be probably preferred if this matrix is to be modified?
//          (double *)this->ObtainWritePartialData(), r, n*(d+1));
    return Y;
  }
  Matrix CartanSyncVariable::mat() const
  {
    // Interface data in the ROPTLIB variable as an Eigen matrix
    Matrix Y = Eigen::Map<Matrix>(
          (double *)this->ObtainReadData(), r, n*(d+1));
    return Y;
  }

  Matrix CartanSyncVariable::R()
  {
    // TODO
  }

  Matrix CartanSyncVariable::R(size_t i)
  {
    return this->mat().block(0,i*(d+1),r,d);
  }

  Matrix CartanSyncVariable::t()
  {
    // TODO
  }

  Matrix CartanSyncVariable::t(size_t i)
  {
    return this->mat().block(0,i*(d+1),r,d);
  }

} /*end of SESync namespace*/
