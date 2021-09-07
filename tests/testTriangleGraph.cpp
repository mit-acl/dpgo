#include <DPGO/PGOAgent.h>

#include "gtest/gtest.h"

using namespace DPGO;

TEST(testDPGO, TriangleGraph) {
  unsigned int id = 0;
  unsigned int d, r;
  d = 3;
  r = 3;
  PGOAgentParameters options(d, r, 1);
  PGOAgent agent(id, options);

  Matrix Tw0(d + 1, d + 1);
  Tw0 << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

  Matrix Tw1(d + 1, d + 1);
  Tw1 << 0.1436, 0.7406, 0.6564, 1, -0.8179, -0.2845, 0.5000, 1, 0.5571,
      -0.6087, 0.5649, 1, 0, 0, 0, 1;

  Matrix Tw2(d + 1, d + 1);
  Tw2 << -0.4069, -0.4150, -0.8138, 2, 0.4049, 0.7166, -0.5679, 2, 0.8188,
      -0.5606, -0.1236, 2, 0, 0, 0, 1;

  Matrix Ttrue(d, 3 * (d + 1));
  Ttrue << 1, 0, 0, 0, 0.1436, 0.7406, 0.6564, 1, -0.4069, -0.4150, -0.8138, 2,
      0, 1, 0, 0, -0.8179, -0.2845, 0.5000, 1, 0.4049, 0.7166, -0.5679, 2, 0, 0,
      1, 0, 0.5571, -0.6087, 0.5649, 1, 0.8188, -0.5606, -0.1236, 2;

  std::vector<RelativeSEMeasurement> odometry;
  std::vector<RelativeSEMeasurement> private_loop_closures;
  std::vector<RelativeSEMeasurement> shared_loop_closures;

  Matrix dT;
  dT = Tw0.inverse() * Tw1;
  RelativeSEMeasurement m01(id, id, 0, 1, dT.block(0, 0, d, d),
                            dT.block(0, d, d, 1), 1.0, 1.0);
  odometry.push_back(m01);

  dT = Tw1.inverse() * Tw2;
  RelativeSEMeasurement m12(id, id, 1, 2, dT.block(0, 0, d, d),
                            dT.block(0, d, d, 1), 1.0, 1.0);
  odometry.push_back(m12);

  dT = Tw0.inverse() * Tw2;
  RelativeSEMeasurement m02(id, id, 0, 2, dT.block(0, 0, d, d),
                            dT.block(0, d, d, 1), 1.0, 1.0);
  private_loop_closures.push_back(m02);

  agent.setPoseGraph(odometry, private_loop_closures, shared_loop_closures);

  Matrix Testimated;
  agent.getTrajectoryInLocalFrame(Testimated);
  ASSERT_LE((Ttrue - Testimated).norm(), 1e-4);

  agent.iterate();

  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.num_poses(), 3);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.relaxation_rank(), r);

  agent.getTrajectoryInLocalFrame(Testimated);
  ASSERT_LE((Ttrue - Testimated).norm(), 1e-4);
}