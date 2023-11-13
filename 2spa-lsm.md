# Proof of Concept: Using 2S-PA for Location Scale Model
2023-11-12

In this note, I use the [Julia](https://julialang.org/) language to
implement the optimization algorithm, mainly for computational speed. It
should be possible to implement this in other languages (e.g., R, SAS)
as well.

``` julia
using DataFrames, XLSX
using DelimitedFiles
using LinearAlgebra, Distributions
using FastGaussQuadrature: gausshermite
using Optim, LineSearches, ForwardDiff
```

# Using Manifest Variables: Replicating Model $A_2$ in Blozis (2022)

## Data Import

``` julia
df = DataFrame(XLSX.readtable("DailyPA.xlsx", "DailyPA"))
# Manifest variables with listwise deletion
mf_df = dropmissing(df)[:, [:ymean_1, :ymean_2, :ymean_3, :ymean_4,
       :ymean_5, :ymean_6, :ymean_7, :ymean_8]]
# Add ID column
mf_df[:, :id] = 1:size(mf_df, 1)
print(first(mf_df, 5))
```

    5×9 DataFrame
     Row │ ymean_1  ymean_2  ymean_3  ymean_4  ymean_5  ymean_6  ymean_7  ymean_8  id    
         │ Any      Any      Any      Any      Any      Any      Any      Any      Int64 
    ─────┼───────────────────────────────────────────────────────────────────────────────
       1 │ 3        3        3        3.8      3        3        2.8      2.8          1
       2 │ 1.8      1.4      1.4      1.4      1.4      1.6      1.6      1.4          2
       3 │ 1.8      2.4      1.2      2.4      3        1.6      2.2      1.8          3
       4 │ 3.2      3        3.2      3.4      2.8      2.8      3.2      3.6          4
       5 │ 3.2      3.6      3.4      3.4      3.2      3.6      3.8      3            5

``` julia
# Convert to long format
mf_dat = Matrix{Float64}(sort(stack(mf_df, Not(:id)), :id)[:, [1, 3]])
# Split data into N array of matrices
mf_arr = [mf_dat[mf_dat[:, 1].==i, :] for i in unique(mf_dat[:, 1])];
```

## G-H Quadrature

``` julia
# Quadrature nodes and weights
gh = gausshermite(7)  # 7 quadrature points (as in Blozis, 2022)
gh_node = gh[1] .* √2
gh_weight = gh[2] ./ √π
DataFrame([gh_node, gh_weight], [:node, :weight])
```

## Likelihood Function

See Appendix A of Blozis (2022).

### Conditional Likelihood

``` julia
function condlikelihood(a, (β₀, βᵤₐ, ϕᵤ_a, τ₀), yᵢ)
    μᵢ_a = β₀ + βᵤₐ * a
    nᵢ = size(yᵢ, 1)
    Σᵢ_a = fill(ϕᵤ_a, nᵢ, nᵢ) + exp(τ₀ + a) * I
    if !isposdef(Σᵢ_a)
        return Inf
    end
    dᵢ = MvNormal(fill(μᵢ_a, nᵢ), Σᵢ_a)
    pdf(dᵢ, yᵢ)
end
```

    condlikelihood (generic function with 1 method)

### Marginal Log-Likelihood

``` julia
# Log-likelihood
function loglikelihood((β₀, α₀, ϕₐ, zρᵤₐ, τ₀), dat, z, w)
    ρᵤₐ = tanh(zρᵤₐ)
    a = z .* ϕₐ
    βᵤₐ = ρᵤₐ * exp(α₀ / 2) / ϕₐ
    ϕᵤ_a = exp(α₀) * (1 - ρᵤₐ^2)
    loglik = 0
    for i in eachindex(dat)
        yᵢ = dat[i][:, 2]
        nᵢ = size(yᵢ, 1)
        ℒᵢ = dot(condlikelihood.(a,
                Ref((β₀, βᵤₐ, ϕᵤ_a, τ₀)), Ref(yᵢ)),
            w)
        loglik += log(ℒᵢ)
    end
    return loglik
end
```

    loglikelihood (generic function with 1 method)

## Optimization

Here I use the Newton’s method with automatic differentiation to find
the MLE. The Fisher’s $Z$ (arctanh) tranformation is applied to
$\rho_{\upsilon a}$ so that the parameter space is unbounded.

``` julia
# Starting values: γ₀ = (β₀, α₀, ϕₐ, zρᵤₐ, τ₀))
γ₀ = [3.0, -1.0, 1.0, 0.0, -1.0]
mf_opt = optimize(x -> -loglikelihood(x, mf_arr, gh_node, gh_weight),
    γ₀, Newton(); autodiff=:forward)
```

     * Status: success (objective increased between iterations)

     * Candidate solution
        Final objective value:     2.559826e+03

     * Found with
        Algorithm:     Newton's Method

     * Convergence measures
        |x - x'|               = 5.75e-08 ≰ 0.0e+00
        |x - x'|/|x'|          = 2.05e-08 ≰ 0.0e+00
        |f(x) - f(x')|         = 1.36e-12 ≰ 0.0e+00
        |f(x) - f(x')|/|f(x')| = 5.33e-16 ≰ 0.0e+00
        |g(x)|                 = 6.65e-12 ≤ 1.0e-08

     * Work counters
        Seconds run:   1  (vs limit Inf)
        Iterations:    6
        f(x) calls:    20
        ∇f(x) calls:   20
        ∇²f(x) calls:  6

``` julia
hess = ForwardDiff.hessian(x -> -loglikelihood(x, mf_arr,
        gh_node, gh_weight), mf_opt.minimizer)
DataFrame([mf_opt.minimizer, sqrt.(diag(inv(hess)))], ["Coef", "SE"])
```

Deviance

``` julia
mf_opt.minimum * 2
```

    5119.651015065089

# 2S-PA Example: Model $B_3$ in Blozis (2022)

The factor score data is obtained with the Bartlett method from a scalar
invariant longitudinal factor model.

## Two-Stage Estimation

Under linear factor models, factor scores ($\tilde {\boldsymbol{\eta}}$)
are linear combinations of item scores,

$$\tilde{\boldsymbol{\eta}}_i = \mathbf{A} \mathbf{y}_i$$

Thus, factor scores can be considered an observed (linear) indicator of
the latent variable(s) ${\boldsymbol{\eta}}$,

$$\tilde{\boldsymbol{\eta}}_i = \mathbf{A} \boldsymbol{\Lambda} \boldsymbol{\eta} + \tilde {\boldsymbol{e}}_i,$$

where $\boldsymbol{e}_{\tilde \eta}$ is the measurement error vector of
the vector scores. For Bartlett scores, it can be shown that
$\mathbf{A} \boldsymbol{\Lambda} = \mathbf{I}$, an identity matrix. The
covariance matrix $\boldsymbol{\Theta}_{\tilde \eta}$ of
$\tilde {\boldsymbol{e}}_i$ can be obtained in standard factor analysis
software (or analytically). When measurement errors are independent in
the original factor model, $\boldsymbol{\Theta}_{\tilde e}$ is a
diagonal matrix.

Note that the above has not considered the mean structure, but for
Bartlett scores, generally the implied measurement intercept is zero.

In the two-stage estimation, one first fits a measurement model and
obtain $\tilde{\boldsymbol{\eta}}$ and $\boldsymbol{\Theta}_{\tilde e}$.
Then, in the second stage, $\tilde{\boldsymbol{\eta}}$ is treated as
single indicators with known variance of measurement errors.

<div>

> **Note**
>
> The estimation is a special case of sec 3.1 of Blozis (2022), with
> $\Lambda$ fixed as an identity matrix and $\Psi$ =
> $\boldsymbol{\Theta}_{\tilde e}$ as fixed parameters.

</div>

<div>

> **Caution**
>
> The second-stage estimation treats the covariance matrix
> $\boldsymbol{\Theta}_{\tilde e}$ as known, but it is an estimate based
> on the first-stage measurement model. When the sample size is large
> and/or the reliability of the factor scores are high, the uncertainty
> in the estimate of $\boldsymbol{\Theta}_{\tilde e}$ is generally
> negligible. Otherwise, a first-order correction on the standard errors
> of the second-stage estimation is possible.

</div>

## Data Import

``` julia
b3_fs = readdlm("b3_fs.dat", ' ')
first(DataFrame(b3_fs,
        [:PA1, :PA2, :PA3, :PA4, :PA5, :PA6, :PA7, :PA8]), 5)
```

``` julia
# Convert to long format (id, eta-tilde)
b3_long = hcat(repeat(1:size(b3_fs, 1), inner=8),
    vec(b3_fs[:, 1:8]'))
# Split data into N array of matrices
b3_arr = [b3_long[b3_long[:, 1].==i, :]
          for i in unique(b3_long[:, 1])];
```

## Without Accounting for Measurement Errors

Here I use 10 quadrature points as in Blozis (2022).

``` julia
# Quadrature nodes and weights
gh = gausshermite(10)  # 10 quadrature points
gh_node = gh[1] .* √2
gh_weight = gh[2] ./ √π
DataFrame([gh_node, gh_weight], [:node, :weight])
```

Here, I treat the factor scores as if they do not contain measurement
error.

``` julia
loglikelihood(γ₀, b3_arr, gh_node, gh_weight)
b3_fs_opt = optimize(x -> -loglikelihood(x, b3_arr, gh_node, gh_weight),
    γ₀, Newton(); autodiff=:forward)
b3_fs_opt.minimizer
```

    5-element Vector{Float64}:
      3.0278057099648237
     -1.2413552921509428
      1.7761123955482534
     -0.3306912796092414
     -2.7117368177934678

``` julia
# Obtain hessian matrix (for SEs)
hess_fs = ForwardDiff.hessian(
    x -> -loglikelihood(x, b3_arr, gh_node, gh_weight),
    b3_fs_opt.minimizer);
```

Table of Coefficients

``` julia
DataFrame([["beta_0", "alpha_0", "phi_a", "zrho_ups, a", "tau_0"],
    round.(b3_fs_opt.minimizer, digits=3),
    round.(sqrt.(diag(inv(hess_fs))), digits=3)],
    ["Parameter", "Coef", "SE"])
```

## Accounting for Measurement Errors

Here is $\boldsymbol{\Theta}_{\tilde e}$ from the measurement model for
the Bartlett scores.

``` julia
# Note: with Bartlett scores and independent errors,# the covariance matrix is diagonal
# Θₑ
b3_cov = Diagonal(readdlm("b3_cov.dat", ' '))
```

    8×8 Diagonal{Float64, Vector{Float64}}:
     0.0514273   ⋅          ⋅         …   ⋅          ⋅          ⋅ 
      ⋅         0.0415078   ⋅             ⋅          ⋅          ⋅ 
      ⋅          ⋅         0.0386135      ⋅          ⋅          ⋅ 
      ⋅          ⋅          ⋅             ⋅          ⋅          ⋅ 
      ⋅          ⋅          ⋅             ⋅          ⋅          ⋅ 
      ⋅          ⋅          ⋅         …  0.0396331   ⋅          ⋅ 
      ⋅          ⋅          ⋅             ⋅         0.0377896   ⋅ 
      ⋅          ⋅          ⋅             ⋅          ⋅         0.0452394

``` julia
# Conditional likelihood
function condlikelihood(a, (β₀, βᵤₐ, ϕᵤ_a, τ₀), Θₑ, yᵢ)
    μᵢ_a = β₀ + βᵤₐ * a
    nᵢ = size(yᵢ, 1)
    Σᵢ_a = fill(ϕᵤ_a, nᵢ, nᵢ) + exp(τ₀ + a) * I + Θₑ
    if !isposdef(Σᵢ_a)
        return Inf
    end
    dᵢ = MvNormal(fill(μᵢ_a, nᵢ), Σᵢ_a)
    pdf(dᵢ, yᵢ)
end
# Log-likelihood
function loglikelihood((β₀, α₀, ϕₐ, zρᵤₐ, τ₀), dat, Θₑ, z, w)
    ρᵤₐ = tanh(zρᵤₐ)
    a = z .* ϕₐ
    βᵤₐ = ρᵤₐ * exp(α₀ / 2) / ϕₐ
    ϕᵤ_a = exp(α₀) * (1 - ρᵤₐ^2)
    loglik = 0
    for i in eachindex(dat)
        yᵢ = dat[i][:, 2]
        nᵢ = size(yᵢ, 1)
        ℒᵢ = dot(condlikelihood.(a,
                Ref((β₀, βᵤₐ, ϕᵤ_a, τ₀)), Ref(Θₑ), Ref(yᵢ)),
            w)
        loglik += log(ℒᵢ)
    end
    return loglik
end
loglikelihood(γ₀, b3_arr, b3_cov, gh_node, gh_weight)
b3_2spa_opt = optimize(x -> -loglikelihood(x, b3_arr, b3_cov,
        gh_node, gh_weight),
    γ₀, Newton(); autodiff=:forward)
```

     * Status: success

     * Candidate solution
        Final objective value:     2.268432e+03

     * Found with
        Algorithm:     Newton's Method

     * Convergence measures
        |x - x'|               = 4.34e-07 ≰ 0.0e+00
        |x - x'|/|x'|          = 1.45e-07 ≰ 0.0e+00
        |f(x) - f(x')|         = 5.55e-11 ≰ 0.0e+00
        |f(x) - f(x')|/|f(x')| = 2.45e-14 ≰ 0.0e+00
        |g(x)|                 = 4.72e-11 ≤ 1.0e-08

     * Work counters
        Seconds run:   1  (vs limit Inf)
        Iterations:    6
        f(x) calls:    19
        ∇f(x) calls:   19
        ∇²f(x) calls:  6

### Standard Errors

By inverting the Hessian matrix

``` julia
hess = ForwardDiff.hessian(x -> -loglikelihood(x, b3_arr, b3_cov,
        gh_node, gh_weight),
    b3_2spa_opt.minimizer)
Vᵧ = inv(hess)
sqrt.(diag(Vᵧ))
```

    5-element Vector{Float64}:
     0.027074174844040486
     0.07388805190585837
     0.08093602930090095
     0.06546199831230831
     0.08996260604253589

First-order correction for SE

$$\hat V_{\gamma, c} = \hat V_{\gamma} + (\boldsymbol{H}_\gamma)^{-1} \left(\frac{\partial^2 \ell}{\partial \theta \partial \gamma^\top}\right) \hat V_{\theta} \left(\frac{\partial^2 \ell}{\partial \theta \partial \gamma^\top}\right)^\top (\boldsymbol{H}_\gamma)^{-1},$$

where $V_{\gamma}$ is the naive covariance matrix of
$\hat{boldsymbol{\gamma}}$ assuming the measurement error variance
parameter, $\boldsymbol{\theta}$, is known, $\boldsymbol{H}_\gamma$ is
the Hessian matrix of the log-likelihood $\ell$ with respect to
$\hat{boldsymbol{\gamma}}$, and $V_{\theta}$ can be obtained in the
first-stage measurement model analysis.

``` julia
# Obtain matrix of mixed partial derivatives
f(γ, θ) = ForwardDiff.gradient(x -> -loglikelihood(x, b3_arr,
        Diagonal(θ), gh_node, gh_weight),
    γ)
Jθ = ForwardDiff.jacobian(x -> f(b3_2spa_opt.minimizer, x),
    diag(b3_cov))
b3_vcov_ev = readdlm("b3_vcov_ev.dat", ' ')
# Corrected Vᵧ (A \ B = inv(A) * B)
Vᵧ_c = Vᵧ + (hess \ Jθ) * b3_vcov_ev * (hess \ Jθ)'
corrected_se = sqrt.(diag(Vᵧ_c));
```

Table of Coefficients

``` julia
DataFrame([["beta_0", "alpha_0", "phi_a", "zrho_ups, a", "tau_0"],
    round.(b3_2spa_opt.minimizer, digits=3),
    round.(sqrt.(diag(Vᵧ)), digits=3),
    round.(corrected_se, digits=3)],
    ["Parameter", "Coef (2S-PA)",
     "Naive SE (2S-PA)", "Corrected SE (2S-PA)"])
```

The coefficients based on 2S-PA and the corrected SE are very close to
the results for Model $B_3$ in Table 2 of Blozis (2022). The correlation
parameter (row 4) and standard errors are also very similar after the
back transformation of Fisher’s $Z$.

### More quadrature points

Here I use 31 quadrature points, and repeat the 2S-PA analysis.

``` julia
# Quadrature nodes and weights
gh = gausshermite(31)  # 31 quadrature points
gh_node = gh[1] .* √2
gh_weight = gh[2] ./ √π
DataFrame([gh_node, gh_weight], [:node, :weight])
```

``` julia
b3_2spa_opt = optimize(x -> -loglikelihood(x, b3_arr, b3_cov,
        gh_node, gh_weight),
    γ₀, Newton(); autodiff=:forward)
hess = ForwardDiff.hessian(x -> -loglikelihood(x, b3_arr, b3_cov,
        gh_node, gh_weight),
    b3_2spa_opt.minimizer)
Vᵧ = inv(hess)
```

    5×5 Matrix{Float64}:
      0.00074274   -8.06527e-5    2.45079e-5    8.27757e-5   -0.000759568
     -8.06527e-5    0.00548147    0.000541767  -0.000956531  -9.7153e-5
      2.45079e-5    0.000541767   0.00706965   -0.000169531  -0.00300314
      8.27757e-5   -0.000956531  -0.000169531   0.00433679   -0.000375682
     -0.000759568  -9.7153e-5    -0.00300314   -0.000375682   0.00780864

``` julia
# Obtain matrix of mixed partial derivatives
f(γ, θ) = ForwardDiff.gradient(x -> -loglikelihood(x, b3_arr,
        Diagonal(θ), gh_node, gh_weight),
    γ)
Jθ = ForwardDiff.jacobian(x -> f(b3_2spa_opt.minimizer, x),
    diag(b3_cov))
b3_vcov_ev = readdlm("b3_vcov_ev.dat", ' ')
# Corrected Vᵧ (A \ B = inv(A) * B)
Vᵧ_c = Vᵧ + (hess \ Jθ) * b3_vcov_ev * (hess \ Jθ)'
corrected_se = sqrt.(diag(Vᵧ_c));
```

Table of Coefficients

``` julia
DataFrame([["beta_0", "alpha_0", "phi_a", "zrho_ups, a", "tau_0"],
    round.(b3_2spa_opt.minimizer, digits=3),
    round.(sqrt.(diag(Vᵧ)), digits=3),
    round.(corrected_se, digits=3)],
    ["Parameter", "Coef (2S-PA)",
     "Naive SE (2S-PA)", "Corrected SE (2S-PA)"])
```

## With Partial Scalar Invariance

``` julia
pscalar_fs = readdlm("pscalar_fs.dat", ' ')
first(DataFrame(pscalar_fs,
        [:PA1, :PA2, :PA3, :PA4, :PA5, :PA6, :PA7, :PA8]), 5)
```

``` julia
# Convert to long format (id, eta-tilde)
pscalar_long = hcat(repeat(1:size(pscalar_fs, 1), inner=8),
    vec(pscalar_fs[:, 1:8]'))
# Split data into N array of matrices
pscalar_arr = [pscalar_long[pscalar_long[:, 1].==i, :]
               for i in unique(pscalar_long[:, 1])];
```

``` julia
# Note: with Bartlett scores and independent errors,# the covariance matrix is diagonal
# Θₑ
pscalar_cov = readdlm("pscalar_cov.dat", ' ')
# Force positive definite
pscalar_cov = Symmetric(pscalar_cov, :L)
```

    8×8 Symmetric{Float64, Matrix{Float64}}:
     0.0520064   0.0101589   0.00733019  …  0.00548514  0.00512004  0.00756768
     0.0101589   0.0379425   0.00790218     0.00348203  0.00489503  0.00631913
     0.00733019  0.00790218  0.0374119      0.00736252  0.00757489  0.00904347
     0.00574844  0.00493516  0.0100619      0.005597    0.00672912  0.00799522
     0.00701523  0.00568059  0.00834594     0.0075974   0.00809174  0.00984275
     0.00548514  0.00348203  0.00736252  …  0.0362879   0.00809799  0.00754994
     0.00512004  0.00489503  0.00757489     0.00809799  0.035717    0.0109228
     0.00756768  0.00631913  0.00904347     0.00754994  0.0109228   0.0437512

``` julia
loglikelihood(b3_2spa_opt.minimizer, pscalar_arr, pscalar_cov,
    gh_node, gh_weight)
pscalar_2spa_opt = optimize(x -> -loglikelihood(x,
        pscalar_arr, pscalar_cov,
        gh_node, gh_weight),
    b3_2spa_opt.minimizer, Newton(); autodiff=:forward)
hess = ForwardDiff.hessian(x -> -loglikelihood(x,
        pscalar_arr, pscalar_cov,
        gh_node, gh_weight),
    pscalar_2spa_opt.minimizer)
Vᵧ = inv(hess)
```

    5×5 Matrix{Float64}:
      0.000680015  -7.41491e-5    2.33403e-5    8.0984e-5    -0.00070793
     -7.41491e-5    0.0057179     0.000523192  -0.000919692  -9.51745e-5
      2.33403e-5    0.000523192   0.00703146   -0.000226693  -0.00280936
      8.0984e-5    -0.000919692  -0.000226693   0.00424027   -0.0003332
     -0.00070793   -9.51745e-5   -0.00280936   -0.0003332     0.00772065

Table of Coefficients

``` julia
DataFrame([["beta_0", "alpha_0", "phi_a", "zrho_ups, a", "tau_0"],
    round.(pscalar_2spa_opt.minimizer, digits=3),
    round.(sqrt.(diag(Vᵧ)), digits=3)],
    ["Parameter", "Coef (2S-PA)", "Naive SE (2S-PA)"])
```

<div>

> **Potential Extensions**
>
> - Fitting Model $B_4$
> - Adjusting for partial invariance and autocorrelated measurement
>   errors in the first stage
> - Simulation to compare joint and two-stage estimations
> - Incorporate IRT/ordinal CFA in the first stage

</div>
