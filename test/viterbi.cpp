#include <gtest/gtest.h>

#include "hmm.hpp"

namespace hmmcpp {

TEST(viterbi, viterbi1) {
  Hmm::DouleMatrix A{
      {0.5, 0.2, 0.3},
      {0.3, 0.5, 0.2},
      {0.2, 0.3, 0.5},
  };

  Hmm::DouleMatrix B{
      {0.5, 0.5},
      {0.4, 0.6},
      {0.7, 0.3},
  };

  Hmm::DouleVector pi{0.2, 0.4, 0.4};

  auto lambda = Hmm(A, B, pi);

  Hmm::IntVector O{0, 1, 0};

  int T = 3;

  Hmm::IntVector q(T, 0);

  Hmm::viterbi(lambda, T, O, &q);

  Hmm::IntVector res{2, 2, 2};

  EXPECT_EQ(q.size(), res.size());
  for (int i = 0; i < q.size(); ++i) {
    EXPECT_EQ(q[i], res[i]);
  }
}

}  // namespace hmmcpp