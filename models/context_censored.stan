// Hierarchical Gaussian log2 MIC model with exact, left, right and interval data.
// Random effects are study/chemical/target associations, not causal components.
data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N,K] X;
  vector[N] lower_bound;
  vector[N] upper_bound;
  array[N] int<lower=1,upper=4> kind;
  int<lower=1> J_chem;
  int<lower=1> J_pub;
  int<lower=1> J_target;
  int<lower=1> J_chem_target;
  int<lower=1> J_chem_medium;
  array[N] int<lower=1,upper=J_chem> chem;
  array[N] int<lower=1,upper=J_pub> pub;
  array[N] int<lower=1,upper=J_target> target_id;
  array[N] int<lower=1,upper=J_chem_target> chem_target;
  array[N] int<lower=1,upper=J_chem_medium> chem_medium;
  real<lower=0> prior_scale;
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
  vector[N] mu = alpha + X*beta
    + tau[1]*z_chem[chem] + tau[2]*z_pub[pub]
    + tau[3]*z_target[target_id] + tau[4]*z_chem_target[chem_target]
    + tau[5]*z_chem_medium[chem_medium];
  alpha ~ normal(3, 5);
  beta ~ normal(0, prior_scale);
  sigma ~ normal(0, 3);
  tau ~ normal(0, prior_scale);
  z_chem ~ std_normal();
  z_pub ~ std_normal();
  z_target ~ std_normal();
  z_chem_target ~ std_normal();
  z_chem_medium ~ std_normal();
  for (n in 1:N) {
    if (kind[n] == 1)
      target += normal_lpdf(lower_bound[n] | mu[n], sigma);
    else if (kind[n] == 2)
      target += normal_lcdf(upper_bound[n] | mu[n], sigma);
    else if (kind[n] == 3)
      target += normal_lccdf(lower_bound[n] | mu[n], sigma);
    else if (lower_bound[n] > mu[n])
      target += log_diff_exp(normal_lccdf(lower_bound[n] | mu[n], sigma),
                             normal_lccdf(upper_bound[n] | mu[n], sigma));
    else
      target += log_diff_exp(normal_lcdf(upper_bound[n] | mu[n], sigma),
                             normal_lcdf(lower_bound[n] | mu[n], sigma));
  }
}
generated quantities {
  vector[6] conditional_variance_fraction;
  real total = square(sigma) + dot_self(tau);
  for (j in 1:5)
    conditional_variance_fraction[j] = square(tau[j]) / total;
  conditional_variance_fraction[6] = square(sigma) / total;
}
