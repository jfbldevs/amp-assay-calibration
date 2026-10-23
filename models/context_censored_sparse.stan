// Expanded-catalog arithmetic: identical fixed predictor using CSR multiplication.
// Same Gaussian random-effects model with singleton effects integrated exactly.
// An effect level used once contributes tau^2 to that observation's variance.
data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N,K] X;
  matrix<lower=0,upper=1>[N,5] singleton;
  vector[N] lower_bound;
  vector[N] upper_bound;
  array[N] int<lower=1,upper=4> kind;
  int<lower=0> N_exact;
  int<lower=0> N_left;
  int<lower=0> N_right;
  int<lower=0> N_interval;
  array[N_exact] int<lower=1,upper=N> exact_index;
  array[N_left] int<lower=1,upper=N> left_index;
  array[N_right] int<lower=1,upper=N> right_index;
  array[N_interval] int<lower=1,upper=N> interval_index;
  int<lower=0> J_chem;
  int<lower=0> J_pub;
  int<lower=0> J_target;
  int<lower=0> J_chem_target;
  int<lower=0> J_chem_medium;
  array[N] int<lower=1,upper=J_chem+1> chem;
  array[N] int<lower=1,upper=J_pub+1> pub;
  array[N] int<lower=1,upper=J_target+1> target_id;
  array[N] int<lower=1,upper=J_chem_target+1> chem_target;
  array[N] int<lower=1,upper=J_chem_medium+1> chem_medium;
  real<lower=0> prior_scale;
}
transformed data {
  int X_nnz = num_elements(csr_extract_w(X));
  vector[X_nnz] X_w = csr_extract_w(X);
  array[X_nnz] int X_v = csr_extract_v(X);
  array[N+1] int X_u = csr_extract_u(X);
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
  vector<lower=0>[5] tau;
  vector[J_chem] z_chem;
  vector[J_pub] z_pub;
  vector[J_target] z_target;
  vector[J_chem_target] z_chem_target;
  vector[J_chem_medium] z_chem_medium;
}
model {
  vector[J_chem+1] u_chem = append_row(0, z_chem);
  vector[J_pub+1] u_pub = append_row(0, z_pub);
  vector[J_target+1] u_target = append_row(0, z_target);
  vector[J_chem_target+1] u_chem_target = append_row(0, z_chem_target);
  vector[J_chem_medium+1] u_chem_medium = append_row(0, z_chem_medium);
  vector[N] observation_sd = sqrt(rep_vector(square(sigma), N) + singleton*square(tau));
  vector[N] mu = alpha + csr_matrix_times_vector(N, K, X_w, X_v, X_u, beta)
    + tau[1]*u_chem[chem] + tau[2]*u_pub[pub]
    + tau[3]*u_target[target_id] + tau[4]*u_chem_target[chem_target]
    + tau[5]*u_chem_medium[chem_medium];
  alpha ~ normal(3, 5);
  beta ~ normal(0, prior_scale);
  sigma ~ normal(0, 3);
  tau ~ normal(0, prior_scale);
  z_chem ~ std_normal();
  z_pub ~ std_normal();
  z_target ~ std_normal();
  z_chem_target ~ std_normal();
  z_chem_medium ~ std_normal();
  target += normal_lpdf(lower_bound[exact_index] | mu[exact_index], observation_sd[exact_index]);
  target += normal_lcdf(upper_bound[left_index] | mu[left_index], observation_sd[left_index]);
  target += normal_lccdf(lower_bound[right_index] | mu[right_index], observation_sd[right_index]);
  if (N_interval > 0) for (i in 1:N_interval) {
    int n = interval_index[i];
    if (lower_bound[n] > mu[n])
      target += log_diff_exp(normal_lccdf(lower_bound[n] | mu[n], observation_sd[n]),
                             normal_lccdf(upper_bound[n] | mu[n], observation_sd[n]));
    else
      target += log_diff_exp(normal_lcdf(upper_bound[n] | mu[n], observation_sd[n]),
                             normal_lcdf(lower_bound[n] | mu[n], observation_sd[n]));
  }
}
generated quantities {
  vector[6] conditional_variance_fraction;
  real total = square(sigma) + dot_self(tau);
  for (j in 1:5)
    conditional_variance_fraction[j] = square(tau[j]) / total;
  conditional_variance_fraction[6] = square(sigma) / total;
}
