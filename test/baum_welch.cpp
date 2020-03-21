#include <gtest/gtest.h>

#include "hmm.hpp"

namespace hmmcpp {

TEST(baum_welch, baum_welch1) {
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

  Hmm lambda_predict(3, 2);

  Hmm::baum_welch(T, O, &lambda_predict);
}

}  // namespace hmmcpp