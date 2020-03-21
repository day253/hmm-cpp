#include <gtest/gtest.h>

#include "hmm.h"

namespace hmmcpp {

// 前向概率
// alpha_t(i) = P(o_1,o_2,...,o_t,i_t = q_i | lambda)
double Hmm::forward(const Hmm& lambda, int T, const IntVector& O) {
  const auto& N = lambda.N;
  const auto& A = lambda.A;
  const auto& B = lambda.B;
  const auto& pi = lambda.pi;

  DouleMatrix alpha(T, DouleVector(N, 0.0));

  // 1. Initialization
  // alpha_1(i) = pi_i * b_i(o_1), i = 1,2,...,N
  // 实际计算从0开始
  for (int i = 0; i < N; ++i) {
    alpha[0][i] = pi[i] * B[i][O[0]];
  }

  // 2. Induction
  // 递推 t = 1,2,...,T - 1
  // alpha_t+1(i) = [sigma^N_(j=1) alpha_t(j) * a_ji] * b_i(o_t+1), i =
  // 1,2,...,N
  for (int t = 0; t < T - 1; ++t) {
    for (int i = 0; i < N; ++i) {
      double sum = 0.0;
      for (int j = 0; j < N; ++j) {
        sum += alpha[t][j] * (A[j][i]);
      }

      alpha[t + 1][i] = sum * (B[i][O[t + 1]]);
    }
  }

  // 3. Termination
  // P(O | lambda) = sigma^N_(i=1) alpha_t(i)
  double prob = 0.0;
  for (int i = 0; i < N; ++i) {
    prob += alpha[T - 1][i];
  }
  return prob;
}

TEST(forward, forward1) {
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

  EXPECT_NEAR(0.13022, Hmm::forward(lambda, 3, O), 1e-5);
}

}  // namespace hmmcpp
