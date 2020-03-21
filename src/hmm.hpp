#pragma once

#include <complex>
#include <vector>

namespace hmmcpp {

class Hmm {
 public:
  typedef std::vector<double> DouleVector;
  typedef std::vector<int> IntVector;
  typedef std::vector<DouleVector> DouleMatrix;
  typedef std::vector<DouleMatrix> DouleCube;

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
  Hmm(int _n, int _m)
      : N(_n),
        M(_m),
        A(DouleMatrix(N, DouleVector(N, 1.0 / N))),
        B(DouleMatrix(N, DouleVector(M, 1.0 / M))),
        pi(DouleVector(N, 1.0 / N)) {}

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
  static double forward_with_scale(const Hmm& lambda, int T,
                                   const IntVector& O);
  static double forward(const Hmm& lambda, int T, const IntVector& O,
                        DouleMatrix* alpha);
  static double forward_with_scale(const Hmm& lambda, int T, const IntVector& O,
                                   DouleMatrix* alpha, DouleVector* scale);
  // 后向概率
  static double backward(const Hmm& lambda, int T, const IntVector& O);
  static double backward_with_scale(const Hmm& lambda, int T,
                                    const IntVector& O);
  static double backward(const Hmm& lambda, int T, const IntVector& O,
                         DouleMatrix* beta);
  static double backward_with_scale(const Hmm& lambda, int T,
                                    const IntVector& O, DouleMatrix* beta,
                                    DouleVector* scale);
  // 维特比解码
  static void viterbi(const Hmm& lambda, int T, const IntVector& O,
                      IntVector* q);
  static void viterbi_log(const Hmm& lambda, int T, const IntVector& O,
                          IntVector* q);
  // EM
  static void compute_gamma(const Hmm& lambda, int T, const DouleMatrix& alpha,
                            const DouleMatrix& beta, DouleMatrix* gamma);
  static void compute_xi(const Hmm& lambda, int T, const IntVector& O,
                         const DouleMatrix& alpha, const DouleMatrix& beta,
                         DouleCube* xi);
  static void baum_welch(int T, const IntVector& O, Hmm* lambda);
  static void baum_welch(int T, const IntVector& O, Hmm* lambda, int* pniter,
                         double* plogprobinit, double* plogprobfinal);
};

double Hmm::forward(const Hmm& lambda, int T, const IntVector& O) {
  const auto& N = lambda.N;
  DouleMatrix alpha(T, DouleVector(N, 0.0));
  return forward(lambda, T, O, &alpha);
}

double Hmm::forward_with_scale(const Hmm& lambda, int T, const IntVector& O) {
  const auto& N = lambda.N;
  DouleMatrix alpha(T, DouleVector(N, 0.0));
  DouleVector scale(T);
  return forward_with_scale(lambda, T, O, &alpha, &scale);
}

// 前向概率
// alpha_t(i) = P(o_1,o_2,...,o_t,i_t = q_i | lambda)
double Hmm::forward(const Hmm& lambda, int T, const IntVector& O,
                    DouleMatrix* alpha) {
  const auto& N = lambda.N;
  const auto& A = lambda.A;
  const auto& B = lambda.B;
  const auto& pi = lambda.pi;
  auto& alpha_ = *alpha;

  // 1. Initialization
  // alpha_1(i) = pi_i * b_i(o_1), i = 1,2,...,N
  // 实际计算从0开始
  for (int i = 0; i < N; ++i) {
    alpha_[0][i] = pi[i] * B[i][O[0]];
  }

  // 2. Induction
  // 递推 t = 1,2,...,T - 1
  // alpha_t+1(i) = [sigma^N_(j=1) alpha_t(j) * a_ji] * b_i(o_t+1), i =
  // 1,2,...,N
  for (int t = 0; t < T - 1; ++t) {
    for (int i = 0; i < N; ++i) {
      double sum = 0.0;
      for (int j = 0; j < N; ++j) {
        sum += alpha_[t][j] * (A[j][i]);
      }

      alpha_[t + 1][i] = sum * (B[i][O[t + 1]]);
    }
  }

  // 3. Termination
  // P(O | lambda) = sigma^N_(i=1) alpha_t(i)
  double prob = 0.0;
  for (int i = 0; i < N; ++i) {
    prob += alpha_[T - 1][i];
  }
  return prob;
}

double Hmm::forward_with_scale(const Hmm& lambda, int T, const IntVector& O,
                               DouleMatrix* alpha, DouleVector* scale) {
  const auto& N = lambda.N;
  const auto& A = lambda.A;
  const auto& B = lambda.B;
  const auto& pi = lambda.pi;
  auto& alpha_ = *alpha;
  auto& scale_ = *scale;

  // 1. Initialization
  // alpha_1(i) = pi_i * b_i(o_1), i = 1,2,...,N
  // 实际计算从0开始
  for (int i = 0; i < N; ++i) {
    alpha_[0][i] = pi[i] * B[i][O[0]];
    scale_[0] += alpha_[0][i];
  }

  for (int i = 0; i < N; ++i) {
    alpha_[0][i] /= scale_[0];
  }

  // 2. Induction
  // 递推 t = 1,2,...,T - 1
  // alpha_t+1(i) = [sigma^N_(j=1) alpha_t(j) * a_ji] * b_i(o_t+1), i =
  // 1,2,...,N
  for (int t = 0; t < T - 1; ++t) {
    scale_[t + 1] = 0.0;
    for (int i = 0; i < N; ++i) {
      double sum = 0.0;
      for (int j = 0; j < N; ++j) {
        sum += alpha_[t][j] * (A[j][i]);
      }

      alpha_[t + 1][i] = sum * (B[i][O[t + 1]]);
      scale_[t + 1] += alpha_[t + 1][i];
    }
    for (int i = 0; i < N; ++i) {
      alpha_[t + 1][i] /= scale_[t + 1];
    }
  }

  // 3. Termination
  // P(O | lambda) = sigma^N_(i=1) alpha_t(i)
  double prob = 0.0;
  for (int t = 0; t < T; ++t) {
    prob += std::log(scale_[t]);
  }
  return prob;
}

double Hmm::backward(const Hmm& lambda, int T, const IntVector& O) {
  const auto& N = lambda.N;
  DouleMatrix beta(T, DouleVector(N, 0.0));
  return backward(lambda, T, O, &beta);
}

double Hmm::backward_with_scale(const Hmm& lambda, int T, const IntVector& O) {
  const auto& N = lambda.N;
  DouleMatrix beta(T, DouleVector(N, 0.0));
  DouleVector scale(T);
  return backward_with_scale(lambda, T, O, &beta, &scale);
}

double Hmm::backward(const Hmm& lambda, int T, const IntVector& O,
                     DouleMatrix* beta) {
  const auto& N = lambda.N;
  const auto& A = lambda.A;
  const auto& B = lambda.B;
  const auto& pi = lambda.pi;
  auto& beta_ = *beta;

  // 1. Initialization
  // beta_T(i) = 1, i = 1,2,...,N
  for (int i = 0; i < N; ++i) {
    beta_[T - 1][i] = 1.0;
  }

  // 2. Induction
  // t = T-1,T-2,...,1
  // beta_t(i) = sigma^N_(j=1) a_ij * b_j(o_t+1)*beta_t+1(j), i = 1,2,...,N
  for (int t = T - 2; t >= 0; --t) {
    for (int i = 0; i < N; ++i) {
      double sum = 0.0;
      for (int j = 0; j < N; ++j) {
        sum += A[i][j] * (B[j][O[t + 1]]) * beta_[t + 1][j];
      }
      beta_[t][i] = sum;
    }
  }

  // 3. Termination
  // P(O | lambda) = sigma^N_(i=1) pi_t * b_i(o_1)*beta_1(i)
  double prob = 0.0;
  for (int i = 0; i < N; ++i) {
    prob += pi[i] * B[i][O[0]] * beta_[0][i];
  }
  return prob;
}

double Hmm::backward_with_scale(const Hmm& lambda, int T, const IntVector& O,
                                DouleMatrix* beta, DouleVector* scale) {
  const auto& N = lambda.N;
  const auto& A = lambda.A;
  const auto& B = lambda.B;
  const auto& pi = lambda.pi;
  auto& beta_ = *beta;
  auto& scale_ = *scale;

  // 1. Initialization
  for (int i = 0; i < N; ++i) {
    beta_[T - 1][i] = 1.0 / scale_[T - 1];
  }

  // 2. Induction
  for (int t = T - 2; t >= 0; --t) {
    for (int i = 0; i < N; ++i) {
      double sum = 0.0;
      for (int j = 0; j < N; ++j) {
        sum += A[i][j] * (B[j][O[t + 1]]) * beta_[t + 1][j];
      }
      beta_[t][i] = sum / scale_[t];
    }
  }

  // 3. Termination
  double prob = 0.0;
  for (int i = 0; i < N; ++i) {
    prob += beta_[0][i];
  }
  return prob;
}

void Hmm::viterbi(const Hmm& lambda, int T, const IntVector& O, IntVector* q) {
  const auto& N = lambda.N;
  const auto& A = lambda.A;
  const auto& B = lambda.B;
  const auto& pi = lambda.pi;
  auto& q_ = *q;

  DouleMatrix delta(T, DouleVector(N, 0.0));
  DouleMatrix psi(T, DouleVector(N, 0.0));

  // 1. Initialization
  for (int i = 0; i < N; ++i) {
    delta[0][i] = pi[i] * (B[i][O[0]]);
    psi[0][i] = 0;
  }

  // 2. Recursion
  for (int t = 1; t < T; ++t) {
    for (int j = 0; j < N; ++j) {
      double maxval = 0.0;
      double maxvalind = 1;
      for (int i = 0; i < N; ++i) {
        double val = delta[t - 1][i] * A[i][j];
        if (val > maxval) {
          maxval = val;
          maxvalind = i;
        }
      }

      delta[t][j] = maxval * (B[j][O[t]]);
      psi[t][j] = maxvalind;
    }
  }

  // 3. Termination
  double prob = 0.0;
  q_[T - 1] = 1;
  for (int i = 0; i < N; ++i) {
    if (delta[T - 1][i] > prob) {
      prob = delta[T - 1][i];
      q_[T - 1] = i;
    }
  }

  // 4. Path (state sequence) backtracking
  for (int t = T - 2; t >= 0; --t) {
    q_[t] = psi[t + 1][q_[t + 1]];
  }
}

void Hmm::viterbi_log(const Hmm& lambda, int T, const IntVector& O,
                      IntVector* q) {
  constexpr double vithuge = -10e30;

  const auto& N = lambda.N;
  auto A = lambda.A;
  const auto& B = lambda.B;
  auto pi = lambda.pi;
  auto& q_ = *q;

  DouleMatrix delta(T, DouleVector(N, 0.0));
  DouleMatrix psi(T, DouleVector(N, 0.0));
  DouleMatrix biot(N, DouleVector(T, 0.0));

  // 0. Preprocessing
  for (int i = 0; i < N; ++i) {
    pi[i] = std::log(pi[i]);
  }
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i][j] = std::log(A[i][j]);
    }
  }
  for (int i = 0; i < N; ++i) {
    for (int t = 0; t < T; ++t) {
      biot[i][t] = std::log(B[i][O[t]]);
    }
  }

  // 1. Initialization
  for (int i = 0; i < N; ++i) {
    delta[0][i] = pi[i] + biot[i][0];
    psi[0][i] = 0;
  }

  // 2. Recursion
  for (int t = 1; t < T; ++t) {
    for (int j = 0; j < N; ++j) {
      double maxval = vithuge;
      double maxvalind = 1;
      for (int i = 0; i < N; ++i) {
        double val = delta[t - 1][i] + A[i][j];
        if (val > maxval) {
          maxval = val;
          maxvalind = i;
        }
      }

      delta[t][j] = maxval + biot[j][O[t]];
      psi[t][j] = maxvalind;
    }
  }

  // 3. Termination
  double prob = vithuge;
  q_[T - 1] = 1;
  for (int i = 0; i < N; ++i) {
    if (delta[T - 1][i] > prob) {
      prob = delta[T - 1][i];
      q_[T - 1] = i;
    }
  }

  // 4. Path (state sequence) backtracking
  for (int t = T - 2; t >= 0; --t) {
    q_[t] = psi[t + 1][q_[t + 1]];
  }
}

void Hmm::compute_gamma(const Hmm& lambda, int T, const DouleMatrix& alpha,
                        const DouleMatrix& beta, DouleMatrix* gamma) {
  const auto& N = lambda.N;
  auto& gamma_ = *gamma;

  // gama_t(i) = P(i_t = q_i | O,lambda)
  //           = P(i_t = q_i,O | lambda) / P(O | lambda)
  // gama_t(i) = alpha_t(i) * beta_t(i) / sigma^N_(j=1) alpha_t(j) * beta_t(j)
  for (int t = 0; t < T; ++t) {
    double denominator = 0.0;
    for (int j = 0; j < N; ++j) {
      gamma_[t][j] = alpha[t][j] * beta[t][j];
      denominator += gamma_[t][j];
    }

    for (int i = 0; i < N; ++i) {
      gamma_[t][i] /= denominator;
    }
  }
}

void Hmm::compute_xi(const Hmm& lambda, int T, const IntVector& O,
                     const DouleMatrix& alpha, const DouleMatrix& beta,
                     DouleCube* xi) {
  const auto& N = lambda.N;
  const auto& A = lambda.A;
  const auto& B = lambda.B;
  auto& xi_ = *xi;

  // xi_t(i,j) = P(i_t = q_i, i_t+1 = q_j | O, lambda)
  //           = P(i_t = q_i, i_t+1 = q_j , O | lambda) / P(O | lambda)
  // xi_t(i,j) = alpha_t(i) * a_ij * b_j(o_t+1) * beta_t+1(j)
  //           / sigma^N_(i=1) sigma^N_(j=1)
  //             alpha_t(i) * a_ij * b_j(o_t+1) * beta_t+1(j)
  for (int t = 0; t < T - 1; ++t) {
    double denominator = 0.0;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        xi_[t][i][j] = alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
        denominator += xi_[t][i][j];
      }
    }

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        xi_[t][i][j] /= denominator;
      }
    }
  }
}

void Hmm::baum_welch(int T, const IntVector& O, Hmm* lambda) {
  int pniter = 0;
  double plogprobinit = 0.0;
  double plogprobfinal = 0.0;
  baum_welch(T, O, lambda, &pniter, &plogprobinit, &plogprobfinal);
}

// 输入
// 观测数据 O=(o1,o2,...,oT)
// 输出
// 隐马尔可夫模型参数
void Hmm::baum_welch(int T, const IntVector& O, Hmm* lambda, int* pniter,
                     double* plogprobinit, double* plogprobfinal) {
  constexpr double threshold = 1e-3;
  constexpr double smooth_add = 1e-3;
  constexpr double smooth_multi = 1 - smooth_add;

  auto& lambda_ = *lambda;
  auto& N = lambda_.N;
  auto& M = lambda_.M;
  auto& A = lambda_.A;
  auto& B = lambda_.B;
  auto& pi = lambda_.pi;

  DouleMatrix alpha(T, DouleVector(N, 0.0));
  DouleMatrix beta(T, DouleVector(N, 0.0));
  DouleMatrix gamma(T, DouleVector(N, 0.0));
  DouleCube xi(T, DouleMatrix(N, DouleVector(N, 0.0)));
  DouleVector scale(T);

  double logprof = forward_with_scale(lambda_, T, O, &alpha, &scale);
  double logprob = backward_with_scale(lambda_, T, O, &beta, &scale);
  compute_gamma(lambda_, T, alpha, beta, &gamma);
  compute_xi(lambda_, T, O, alpha, beta, &xi);

  // log P(O | intial model)
  *plogprobinit = logprof;
  double logprobprev = logprob;

  int l = 0;
  double delta;

  // a^(n+1)_ij = sigma^(T-1)_(t=1) xi_t(i,j) / sigma^(T-1)_(t=1) gamma_t(i)
  // b^(n+1)_j(k) =
  //    sigma^T_(t=1,o_t=v_k) gamma_t(j) / sigma^(T)_(t=1) gamma_t(i)
  // pi^(n+1)_i = gamma_1(i)
  do {
    // reestimate frequency of state i in time t=1
    for (int i = 0; i < N; ++i) {
      pi[i] = smooth_add + smooth_multi * gamma[0][i];
    }

    // reestimate transition matrix and symbol prob in each state
    for (int j = 0; j < N; ++j) {
      double denominatorA = 0.0;
      for (int t = 0; t < T - 1; ++t) {
        denominatorA += gamma[t][j];
      }

      for (int i = 0; i < N; ++i) {
        double numeratorA = 0.0;

        for (int t = 0; t < T - 1; ++t) {
          numeratorA += xi[t][i][j];
        }

        A[i][j] = smooth_add + smooth_multi * numeratorA / denominatorA;
      }

      double denominatorB = denominatorA + gamma[T - 1][j];
      for (int k = 0; k < M; ++k) {
        double numeratorB = 0.0;

        for (int t = 0; t < T; ++t) {
          if (O[t] == k) {
            numeratorB += gamma[t][j];
          }
        }

        B[j][k] = smooth_add + smooth_multi * numeratorB / denominatorB;
      }
    }

    double logprof = forward_with_scale(lambda_, T, O, &alpha, &scale);
    double logprob = backward_with_scale(lambda_, T, O, &beta, &scale);
    compute_gamma(lambda_, T, alpha, beta, &gamma);
    compute_xi(lambda_, T, O, alpha, beta, &xi);

    // compute difference between log probability of two iterations
    delta = logprob - logprobprev;
    logprobprev = logprob;
    ++l;
    // if log probability does not change much, exit
  } while (delta > threshold);

  *pniter = l;
  // log P(O | estimated model)
  *plogprobfinal = logprobprev;
}

}  // namespace hmmcpp