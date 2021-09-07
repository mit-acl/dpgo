#include <DPGO/PGOAgent.h>

#include "gtest/gtest.h"

using namespace DPGO;

TEST(testDPGO, LineGraph) {
  unsigned int id = 0;
  unsigned int d, r;
  d = 3;
  r = 3;
  PGOAgentParameters options(d, r, 1);

  Matrix R = Matrix::Identity(d, d);
  Matrix t = Matrix::Random(d, 1);

  std::vector<RelativeSEMeasurement> odometry;
  std::vector<RelativeSEMeasurement> private_loop_closures;
  std::vector<RelativeSEMeasurement> shared_loop_closures;
  PGOAgent agent(id, options);
  for (unsigned int i = 0; i < 4; ++i) {
    RelativeSEMeasurement m(id, id, i, i + 1, R, t, 1.0, 1.0);
    odometry.push_back(m);
  }
  agent.setPoseGraph(odometry, private_loop_closures, shared_loop_closures);
  agent.iterate();

  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.num_poses(), 5);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.relaxation_rank(), r);
}