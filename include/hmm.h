#pragma once

#include <vector>

namespace hmmcpp {

class Hmm {
 public:
  typedef std::vector<double> DouleVector;
  typedef std::vector<int> IntVector;
  typedef std::vector<DouleVector> DouleMatrix;

  // 状态集合
  // number of states;
  // Q={q_1,q_2,...,q_N}
  int N = 0;

  // 观测集合
  // number of observation symbols;
  // V={v_1,v_2,...,v_M}
  int M = 0;

  // I
  // I=(i_1,i_2,...,i_T)
  // 状态序列

  // O
  // O=(o_1,o_2,...,o_T)
  // 观测序列

  // A[1..N][1..N];
  // 状态转移概率矩阵
  // a_ij = P(i_t+1 = q_j | i_t = q_i)
  // i = 1,2,...,N; j = 1,2,...,N
  // a[i][j] is the transition prob of going from state i at time t to state j
  // at time t+1
  DouleMatrix A;

  // B[1..N][1..M];
  // 观测概率矩阵
  // b_j(k) = P(o_t = v_k | i_t = q_j)
  // k = 1,2,...,M; j = 1,2,...,N
  // b[j][k] is the probability of observing symbol k in state j
  DouleMatrix B;

  // pi[1..N];
  // 初始状态概率向量
  // pi_i = P(i_1 = q_i)
  // i = 1,2,...,N;
  // pi[i] is the initial state distribution.
  DouleVector pi;

 public:
  Hmm(int _m, int _n)
      : M(_m),
        N(_n),
        A(DouleMatrix(N, DouleVector(N, 0.0))),
        B(DouleMatrix(N, DouleVector(M, 0.0))),
        pi(DouleVector(N, 0.0)) {}

  Hmm(const DouleMatrix& _A, const DouleMatrix& _B, const DouleVector& _pi)
      : A(_A), B(_B), pi(_pi) {
    N = A.size();
    if (!B.empty()) {
      M = B[0].size();
    }
  }

  // 前向概率
  // alpha_t(i) = P(o_1,o_2,...,o_t,i_t = q_i | lambda)
  static double forward(const Hmm& lambda, int T, const IntVector& O);
};

}  // namespace hmmcpp