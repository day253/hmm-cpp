#include <gtest/gtest.h>

#include "hmm.hpp"

namespace hmmcpp {

TEST(backward, backward1) {
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

  EXPECT_NEAR(0.13022, Hmm::backward(lambda, T, O), 1e-5);
  // EXPECT_NEAR(0.13022, Hmm::backward_with_scale(lambda, T, O), 1e-5);
}

}  // namespace hmmcpp