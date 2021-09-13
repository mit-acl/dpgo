#include <DPGO/PGOAgent.h>

#include "gtest/gtest.h"

using namespace DPGO;

TEST(testDPGO, Construction) {
  unsigned int id = 1;
  unsigned int d, r;
  d = 3;
  r = 3;
  PGOAgentParameters options(d, r, 1);

  PGOAgent agent(id, options);

  ASSERT_EQ(agent.getID(), id);
  ASSERT_EQ(agent.num_poses(), 1);
  ASSERT_EQ(agent.dimension(), d);
  ASSERT_EQ(agent.relaxation_rank(), r);
}