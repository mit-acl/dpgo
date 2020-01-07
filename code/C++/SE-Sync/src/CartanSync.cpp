#include <chrono>
#include <functional>

#include "SESync.h"
#include "CartanSyncProblem.h"
#include "CartanSyncVariable.h"
#include "SESync_types.h"
#include "SESync_utils.h"

#include "RTRNewton.h"
#include "SolversLS.h"

// Define some macros for chrono
// NOTE: To avoid redefinition conflicts, allocate into separate code block
#define CHRONO_START \
  auto start_time = std::chrono::high_resolution_clock::now();
#define CHRONO_END \
  auto counter = std::chrono::high_resolution_clock::now() - start_time; \
  double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(counter).count() / 1000.0;\
  if (options.verbose) \
    std::cout << "elapsed computation time: " << elapsed_time << " seconds" \
    << std::endl;

namespace SESync {

/** Helper function; trigger the stopping criterion based upon the gradient norm
 * tolerance stored in the SESyncProblem object*/
bool gradient_norm_stopping_criterion_CartanSync(
    ROPTLIB::Variable *x, ROPTLIB::Vector *gf,
    double f, double ngf, double ngf0,
    const ROPTLIB::Problem *problem,
    const ROPTLIB::Solvers *solver) {
  const CartanSyncProblem *SESync_problem_ptr =
      static_cast<const CartanSyncProblem *>(problem);

  return ((ngf < SESync_problem_ptr->RTR_gradient_norm_tolerance) ||
          (((solver->GetPreviousIterateVal() - f) /
                (fabs(solver->GetPreviousIterateVal()) + 1e-6) <
            solver->Tolerance) &&
           (solver->GetIter() > 1)));
}

SESyncResult CartanSync(const std::vector<RelativePoseMeasurement> &measurements,
                        const SESyncOpts &options, const Eigen::MatrixXd &Y0) {

  /// ALGORITHM DATA
  SparseMatrix ConIncT;     // Oriented Connection Incidence matrix for SE(d) measurements
  DiagonalMatrix ConOmegaT; // Connection Precision matrix for SE(d) measurements
  SparseMatrix ConLapT;     // Connection Laplacian for SE(d) measurements

  Matrix Y;             // The current iterate in the Riemannian Staircase

  SparseMatrix B1, B2, B3; // The measurement matrices B1, B2, B3 defined in
                           // equations (69) of the tech report

  SESyncResult results;
  results.status = RS_ITER_LIMIT;

  if (options.verbose) {
    std::cout << "========= Cartan-Sync ==========" << std::endl << std::endl;

    std::cout << "ALGORITHM SETTINGS:" << std::endl << std::endl;
    std::cout << "SE-Sync settings:" << std::endl;
    std::cout << " Initial level of Riemannian staircase: " << options.r0
              << std::endl;
    std::cout << " Maximum level of Riemannian staircase: " << options.rmax
              << std::endl;
    std::cout << " Relative tolerance for minimum eigenvalue computation in "
                 "optimality verification: "
              << options.eig_comp_tol << std::endl;
    std::cout << " Number of Lanczos vectors to use in minimum eigenvalue "
                 "computation: "
              << options.num_Lanczos_vectors << std::endl;
    std::cout << " Maximum number of iterations for eigenvalue computation: "
              << options.max_eig_iterations << std::endl;
    std::cout << " Tolerance for accepting an eigenvalue as numerically "
                 "nonnegative in optimality verification: "
              << options.min_eig_num_tol << std::endl;
    std::cout << " Using " << (options.use_Cholesky ? "Cholseky" : "QR")
              << " decomposition to compute orthogonal projections"
              << std::endl;
    std::cout << " Initialization method: "
              << (options.use_chordal_initialization ? "chordal" : "random")
              << std::endl
              << std::endl;

    std::cout << "ROPTLIB settings:" << std::endl;
    std::cout << " Using preconditioning in RTR: "
              << options.use_preconditioning << std::endl;
    std::cout << " Stopping tolerance for norm of Riemannian gradient: "
              << options.tolgradnorm << std::endl;
    std::cout << " Stopping tolerance for relative function decrease: "
              << options.rel_func_decrease_tol << std::endl;
    std::cout << " Maximum number of trust-region iterations: "
              << options.max_RTR_iterations << std::endl;
    std::cout << " Maximum number of truncated conjugate gradient iterations "
                 "per outer iteration: "
              << options.max_tCG_iterations << std::endl
              << std::endl;
  }

  /// ALGORITHM START
  auto SESync_start_time = std::chrono::high_resolution_clock::now();

  /// INITIALIZATION
  if (options.verbose) {
    std::cout << "INITIALIZATION:" << std::endl;
    std::cout << " Constructing auxiliary data matrices ..." << std::endl;
  }

  // Construct oriented SE(d) incidence and precision matrices
  {
    if (options.verbose)
      std::cout << " Constructing oriented SE(d) connection incidence matrix A ... ";
    CHRONO_START
    construct_oriented_connection_incidence_matrix_T(measurements, ConIncT, ConOmegaT);
    CHRONO_END
  }

  // Construct SE(d) connection Laplacian
  {
    if (options.verbose)
      std::cout << " Constructing SE(d) connection Laplacian Q ... ";
    CHRONO_START
    ConLapT = construct_connection_Laplacian_T(ConIncT, ConOmegaT);
    CHRONO_END
  }

  // Get dimensions
  size_t d = (!measurements.empty() ? measurements[0].t.size() : 0);
  size_t m = measurements.size();
  size_t n = ConLapT.rows()/(d+1);

  // Construct measurement matrices B1, B2, B3
  if (options.verbose)
    std::cout << " Constructing measurement matrices B1, B2, B3 ... ";
  auto B_start_time = std::chrono::high_resolution_clock::now();
  construct_B_matrices(measurements, B1, B2, B3);
  auto B_counter = std::chrono::high_resolution_clock::now() - B_start_time;
  double B_elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(B_counter).count() /
      1000.0;
  if (options.verbose)
    std::cout << "elapsed computation time: " << B_elapsed_time << " seconds"
              << std::endl
              << std::endl;

  /// Construct SESync problem instance

  if (options.verbose)
    std::cout << "Constructing SE-Sync problem instance ... ";

  auto problem_construction_start_time =
      std::chrono::high_resolution_clock::now();
  CartanSyncProblem problem(ConLapT, n,m,d,
                            options.use_Cholesky, options.use_preconditioning);
  problem.RTR_gradient_norm_tolerance = options.tolgradnorm;
  auto problem_construction_counter =
      std::chrono::high_resolution_clock::now() -
      problem_construction_start_time;
  auto problem_construction_elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          problem_construction_counter)
          .count() /
      1000.0;
  if (options.verbose)
    std::cout << "elapsed computation time: "
              << problem_construction_elapsed_time << " seconds" << std::endl
              << std::endl;

  // Set initial relaxation rank
  problem.set_relaxation_rank(options.r0);

  /// Construct initial iterate

  if (Y0.size() != 0) {
    if (options.verbose)
      std::cout << "Using user-supplied initial iterate Y0" << std::endl;

    // The user supplied an initial iterate, so check that it has the correct
    // size, and if so, use this
    assert((Y0.rows() == options.r0) &&
           (Y0.cols() == (problem.dimension()+1) * problem.num_poses()));
    // TODO: Also check the point is feasible (rotation blocks, etc.)

    Y = Y0;
  } else {
    if (options.use_chordal_initialization) {
      if (options.verbose)
        std::cout << "Computing chordal initialization ... ";

      auto chordal_init_start_time = std::chrono::high_resolution_clock::now();
      // Matrix Rinit = chordal_initialization(LGrho,
      // problem.dimension(),options.max_eig_iterations, 100 *
      // options.eig_comp_tol);
      Matrix Rinit = chordal_initialization(problem.dimension(), B3);
      auto chordal_init_counter =
          std::chrono::high_resolution_clock::now() - chordal_init_start_time;
      double chordal_init_elapsed_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              chordal_init_counter)
              .count() /
          1000.0;
      if (options.verbose)
        std::cout << "elapsed computation time: " << chordal_init_elapsed_time
                  << " seconds" << std::endl;

      // Recover translation component as well for Cartan-Sync
      Matrix tinit = recover_translations(B1, B2, Rinit);

      size_t d = problem.dimension();
      size_t n = problem.num_poses();
      Y.resize(options.r0, n*(d+1));
      Y.setZero();
      for (size_t i=0; i<n; i++)
      {
        Y.block(0,i*(d+1),  d,d) = Rinit.block(0,i*d,d,d);
        Y.block(0,i*(d+1)+d,d,1) = tinit.block(0,i,d,1);
      }
    } else {
      if (options.verbose)
        std::cout << "Sampling a random point on the manifold" << std::endl;

      // Generate random point in the manifold
      CartanSyncVariable Yinit(options.r0,problem.dimension(),problem.num_poses());
      Yinit.RandInManifold();

      Y.resize(options.r0, problem.dimension() * problem.num_poses());
      CartanProd2Mat(Yinit, Y);
    }
  }

  auto initialization_counter =
      std::chrono::high_resolution_clock::now() - SESync_start_time;
  double initialization_elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          initialization_counter)
          .count() /
      1000.0;

  results.initialization_time = initialization_elapsed_time;

  if (options.verbose)
    std::cout << "SE-Sync initialization finished; elapsed time: "
              << initialization_elapsed_time << " seconds" << std::endl
              << std::endl;

  if (options.verbose) {
    // Compute and display the initial objective value
    // TODO: Directly call cost function?
    double F0 = 0.5*(Y * ConLapT * Y.transpose()).trace();
    std::cout << "Initial objective value: " << F0;
  }

  /// RIEMANNIAN STAIRCASE
  auto riemannian_staircase_start_time =
      std::chrono::high_resolution_clock::now();
  // Declare pointer to constant ROPTLIB variable type,
  // you cannot use this pointer to change the value being pointed to
//  const CartanSyncVariable *Yopt_ropt;

  integer r; // define outside of for scope to keep its value after exec
  for (r = options.r0; r <= options.rmax; r++) {
//  for (unsigned int r = options.r0; r <= options.rmax; r++) {

    // The elapsed time from the start of the Riemannian Staircase algorithm
    // until the start of this iteration of RTR
    double RTR_iteration_start_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() -
            riemannian_staircase_start_time)
            .count() /
        1000.0;

    if (options.verbose)
      std::cout << std::endl
                << std::endl
                << "====== RIEMANNIAN STAIRCASE (level r = " << r
                << ") ======" << std::endl
                << std::endl;

    // Update rank of relaxation
    problem.set_relaxation_rank(r);

    // Allocate storage for a new iterate
    // TODO: I'm moving this outside the loop to see if memory is preserved
    CartanSyncVariable Yinit_ropt(r,problem.dimension(),problem.num_poses());
    Yinit_ropt.NewMemoryOnWrite();

    // Initialize the value
    Mat2CartanProd(Y, Yinit_ropt);

    /// Set up RTR solver!
    ROPTLIB::RTRNewton RTR(&problem, &Yinit_ropt);

    // Set stopping criteria
    RTR.Stop_Criterion = ROPTLIB::StopCrit::FUN_REL;
    RTR.Tolerance = options.rel_func_decrease_tol;
    // Note that this custom stopping criterion is called before, and IN
    // ADDITION TO, the relative function decrease tolerance; thus, by setting
    // both, we enforce stopping base both upon gradient norm AND relative
    // function decrease

    RTR.StopPtr = &gradient_norm_stopping_criterion_CartanSync;
    RTR.Max_Iteration = options.max_RTR_iterations;
    RTR.maximum_Delta = 1e4;
    RTR.Debug = (options.verbose ? ROPTLIB::DEBUGINFO::ITERRESULT
                                 : ROPTLIB::DEBUGINFO::NOOUTPUT);

    /// RUN RTR!
    RTR.Run();

    // Extract the results
    const CartanSyncVariable *Yopt_ropt = static_cast<const CartanSyncVariable *>(RTR.GetXopt());
//    Yopt_ropt = static_cast<const CartanSyncVariable *>(RTR.GetXopt());

    results.Yopt.resize(r, problem.num_poses() * problem.dimension());
    CartanProd2Mat(*Yopt_ropt, results.Yopt);
    results.SDPval = RTR.Getfinalfun();
    results.gradnorm = RTR.Getnormgf();

    // Record some interesting info about the solving process

    // Obtained function values
    results.function_values.insert(results.function_values.end(),
                                   RTR.GetfunSeries(),
                                   RTR.GetfunSeries() + RTR.GetlengthSeries());

    // Obtained gradient norm values
    results.gradient_norm_values.insert(
        results.gradient_norm_values.end(), RTR.GetgradSeries(),
        RTR.GetgradSeries() + RTR.GetlengthSeries());

    // Elapsed time since the start of the Riemannian Staircase at which these
    // values were obtained
    std::vector<double> RTR_iteration_function_times(
        RTR.GettimeSeries(), RTR.GettimeSeries() + RTR.GetlengthSeries());
    for (unsigned int i = 0; i < RTR_iteration_function_times.size(); i++)
      RTR_iteration_function_times[i] += RTR_iteration_start_time;
    results.elapsed_optimization_times.insert(
        results.elapsed_optimization_times.end(),
        RTR_iteration_function_times.begin(),
        RTR_iteration_function_times.end());

    if (options.verbose) {
      // Display some output to the user
      std::cout << std::endl
                << "Found first-order critical point with value F(Y) = "
                << results.SDPval
                << "!  Elapsed computation time: " << RTR.GetComTime()
                << " seconds" << std::endl
                << std::endl;
      std::cout << "Checking second order optimality ... " << std::endl;
    }

    // Compute the minimum eigenvalue lambda and corresponding eigenvector of Q
    // - Lambda
    auto eig_start_time = std::chrono::high_resolution_clock::now();
    bool eigenvalue_convergence = problem.compute_Q_minus_Lambda_min_eig(
        results.Yopt, results.lambda_min, results.v_min,
        options.max_eig_iterations, options.eig_comp_tol,
        options.num_Lanczos_vectors);
    auto eig_counter =
        std::chrono::high_resolution_clock::now() - eig_start_time;
    double eig_elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(eig_counter)
            .count() /
        1000.0;

    /// Tests for eigenvalue precision
    if (!eigenvalue_convergence) {
      std::cout << "WARNING!  EIGENVALUE COMPUTATION DID NOT CONVERGE TO "
                   "DESIRED PRECISION!"
                << std::endl;
      results.status = EIG_IMPRECISION;
      break;
    }

    // Test nonnegativity of minimum eigenvalue
    if (results.lambda_min > -options.min_eig_num_tol) {
      // results.Yopt is a second-order critical point!
      if (options.verbose)
        std::cout << "Found second-order critical point! (minimum eigenvalue = "
                  << results.lambda_min
                  << "). Elapsed computation time: " << eig_elapsed_time
                  << " seconds" << std::endl;
      results.status = GLOBAL_OPT;
      break;
    } else {

      /// ESCAPE FROM SADDLE!
      ///
      /// We will perform a backtracking line search along the direction of the
      /// negative eigenvector to escape from the saddle point

      // Set the initial step length to 100 times the distance needed to arrive
      // at a trial point whose gradient is large enough to avoid triggering the
      // RTR gradient norm tolerance stopping condition, according to the local
      // second-order model
      double alpha = 2 * 100 * options.tolgradnorm / fabs(results.lambda_min);
      double alpha_min = 1e-6; // Minimum stepsize

      if (options.verbose) {
        std::cout << "Saddle point detected (minimum eigenvalue = "
                  << results.lambda_min
                  << "). Elapsed computation time: " << eig_elapsed_time
                  << " seconds" << std::endl;

        std::cout << "Computing escape direction ... " << std::endl;
      }

      /**  lambda_min is a negative eigenvalue of Q - Lambda, so the KKT
 * conditions for the semidefinite relaxation are not satisfied; this
 * implies that Y is a saddle point of the rank-restricted
 * semidefinite optimization.  Fortunately, the eigenvector v_min
 * corresponding to lambda_min can be used to provide a descent
 * direction from this saddle point, as described in Theorem 3.9 of the
 * paper "A Riemannian Low-Rank Method for Optimization over Semidefinite
 * Matrices with Block-Diagonal Constraints". Define the vector Ydot :=
 * e_{r+1} * v'; this is tangent to the manifold at Y,
 * and provides a direction of negative curvature */

      // Construct an initial estimate in the next "Riemannian staircase" step
      // by lifting the current estimate
      CartanSyncVariable Xlift(r+1, problem.dimension(), problem.num_poses());
      Yopt_ropt->lift( Xlift );
//      Y = Eigen::MatrixXd::Zero(r + 1, problem.dimension() * problem.num_poses());
//      Y.topRows(r) = results.Yopt;

      // Set the tangent vector with negative curvature (descending direction)
      Eigen::MatrixXd Ydot = Eigen::MatrixXd::Zero(
          r + 1, problem.num_poses() * (problem.dimension()+1));
      Ydot.bottomRows<1>() = results.v_min.transpose();

      // Update the rank of the relaxation
      problem.set_relaxation_rank(r + 1);

      // Allocate new variables of the appropriate size
      CartanSyncVariable EtaProd(r+1, problem.dimension(), problem.num_poses());
      CartanSyncVariable Xtest(r+1, problem.dimension(), problem.num_poses());

      // Initialize line search
      bool escape_success = false;
      do {
        alpha /= 2;

        // Retract along the given tangent vector using the given stepsize
        Mat2CartanProd(alpha * Ydot, EtaProd);
        problem.GetDomain()->Retraction(&Xlift, &EtaProd, &Xtest);
        Eigen::MatrixXd YtestMat;
        CartanProd2Mat(Xtest, YtestMat);

        // Ensure that the trial point Ytest has a lower function value than the
        // current iterate Xlift, and that the gradient at Ytest is sufficiently
        // negative that we will not automatically trigger the gradient
        // tolerance stopping criterion at the next iteration
        double FYtest = problem.f(&Xtest);
        double FYtest_gradnorm =
            (ConLapT * YtestMat.transpose()).norm();

        if ((FYtest < results.SDPval) &&
            (FYtest_gradnorm > options.tolgradnorm))
          escape_success = true;
      } while (!escape_success && (alpha > alpha_min));
      if (escape_success) {
        // Update initialization point for next level in the Staircase
        CartanProd2Mat(Xtest, Y);
      } else {
        std::cout << "WARNING!  BACKTRACKING LINE SEARCH FAILED TO ESCAPE FROM "
                     "SADDLE POINT!"
                  << std::endl;
        results.status = SADDLE_POINT;
        break;
      }
    } // if (saddle point)
  }   // Riemannian Staircase

  /// POST-PROCESSING

  if (options.verbose) {
    std::cout << std::endl
              << std::endl
              << "===== END RIEMANNIAN STAIRCASE =====" << std::endl
              << std::endl;

    switch (results.status) {
    case GLOBAL_OPT:
      std::cout << "Found global optimum!" << std::endl;
      break;
    case EIG_IMPRECISION:
      std::cout << "WARNING: Minimum eigenvalue computation did not achieve "
                   "sufficient accuracy; solution may not be globally optimal!"
                << std::endl;
      break;
    case SADDLE_POINT:
      std::cout << "WARNING: Line-search was unable to escape saddle point!"
                << std::endl;
      break;
    case RS_ITER_LIMIT:
      std::cout << "WARNING:  Riemannian Staircase reached the maximum level "
                   "before finding global optimum!"
                << std::endl;
      break;
    }
  }

  // TODO: For simplicity, we put all recovery code here,
  // but this should be refactored into its own method recover_solution,
  // or unlift_point (better! as inverse of lift_point)
  CartanSyncVariable Yopt_feas(problem.dimension(),problem.dimension(),problem.num_poses());

  // Recover optimal solution as variable
  // TODO: Look for better way to do this, by copying variable out of for scope?
  CartanSyncVariable Yopt_ropt(r,problem.dimension(),problem.num_poses());
  Mat2CartanProd(results.Yopt,Yopt_ropt);

  // TODO: VERY probably allocated memory is corrupted,
  // so a hard copy should be done from local variable to global results
  // inside the for loop
  Yopt_ropt.unlift( Yopt_feas );
  //Yopt_ropt->unlift( Yopt_feas );

  // Check orientation (and fix if necessary)
  if ( is_solution_reflected(Yopt_feas) )
  {
    // Swap sign in last row (equivalent to multiplication by reflection)
    Yopt_feas.mat().bottomRows<1>() = -Yopt_feas.mat().bottomRows<1>();
  }

  // TODO: Project to feasible domain using retraction of the variable
  // Project each block R_i to SO(d)
  for (unsigned int i = 0; i < problem.num_poses(); i++)
    Yopt_feas.R(i) = project_to_SOd( Yopt_feas.R(i) );
  // Or use retraction: Mani->Retraction(x1, eta2, x2);

  {
    Eigen::MatrixXd Y = Yopt_feas.mat();
    results.Fxhat = (Y * ConLapT * Y.transpose()).trace();
  }


  auto SESync_counter =
      std::chrono::high_resolution_clock::now() - SESync_start_time;
  double SESync_elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(SESync_counter)
          .count() /
      1000.0;

  if (options.verbose) {
    std::cout << "Value of SDP solution F(Y): " << results.SDPval << std::endl;
    std::cout << "Norm of Riemannian gradient grad F(Y): " << results.gradnorm
              << std::endl;
    std::cout << "Minimum eigenvalue of Q - Lambda(Y): " << results.lambda_min
              << std::endl;
    std::cout << "Value of rounded pose estimate Rhat: " << results.Fxhat
              << std::endl;
    std::cout << "Suboptimality bound of recovered pose estimate: "
              << results.Fxhat - results.SDPval << std::endl;
    std::cout << "Total elapsed computation time: " << SESync_elapsed_time
              << std::endl
              << std::endl;

    std::cout << "===== END SE-SYNC =====" << std::endl << std::endl;
  }
  return results;
}
}
