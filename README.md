# Pun.jl

**This is a work-in-progress implementation.**

*Automatic Radon-Nikodym Differentiation via **P**robabilistic **Un**computation*

Pun is a new probabilistic programming language, with two distinguishing features:

1. In addition to the usual *assignment operator* (e.g., `x <<= normal(0, 1)`), which adds a random variable to a model, Pun features an *unassignment operator* (written `x >>= normal(0, 1)`), which *marginalizes* a variable from a model. The right-hand side of an unassignment operator is a Pun expression meant to approximate the posterior distribution of that value, given all other variables currently in scope.

2. Pun can automatically compute or estimate the *marginal density functions* of probabilistic programs, with respect to appropriate reference measures.

For example, consider the program

```julia
using Pun

uniform(a, b) = @prob begin
  u <<= random()
  x <<= dirac(u * (b - a) + a)
  u >>= dirac((x - a) / (b - a))
  return x
end

flip(p) = @prob begin
  u <<= random()
  b <<= dirac(u < p)
  u >>= b ? uniform(0, p) : uniform(p, 1)
  return b
end
```

Here, `random` and `dirac` are primitives, corresponding to the distribution $\textsf{Uniform}(0,1)$ and the Dirac delta distribution, respectively. Using these primitives, we define the uniform distribution on a real interval (`uniform`) and the Bernoulli distribution (`flip`). In both cases, our programs generate a random number between 0 and 1, then transform it into the desired random value. They then **unassign** the original random draw, before returning the value of interest. (Pun requires the user to return all variables still in scope at the end of a function's body; if we had not unassigned `u`, we could not return `b` or `x` without also including `u`. Until you marginalize `u`, you have defined a *joint distribution* on both `u` and `x` or `b`.)

We can now evaluate or estimate the density of a particular outcome under our program,
using the `assess` method. In addition to a program and a value in its support, `assess`
expects a *tangent space basis*: viewing the value as a point on a manifold embedded in
$\mathbb{R}^n$, what is a basis of its tangent space at that point? For discrete distributions,
`DiscreteBase` is a zero-dimensional basis provided by the library:

```julia
julia> using Pun: DiscreteBase

julia> assess(flip(0.4), true, DiscreteBase)
-0.916290731874155     # log(0.4)
```

For continuous distributions, use the `LebesgueBase` of the appropriate dimension:

```julia
julia> using Pun: LebesgueBase

julia> assess(uniform(0, 2), 1.2, LebesgueBase(1))
-0.6931471805599453    # log(0.5)
```

In both these cases, the density we get is exact. This is because for every unassignment we performed, we unassigned a variable to its exact posterior distribution, given other variables in scope:
- In `uniform(a, b)`, the posterior of `u` given `x` is precisely `dirac((x-a)/(b-a))`.
- In `flip(p)`, the posterior of `u` given `b` is precisely `uniform(0, p)` when `b` is true, and `uniform(p, 1)` when `b` is false.

In general, Pun will give unbiased density estimates even when unassignments are not exact. However, each unassignment adds variance to the overall density estimator Pun implements, based on the $\chi^2$ divergence from the true posterior to the unassignment distribution.

## Getting Started

### Installation

Pun is a standard Julia package. From the package manager, run

```julia
julia> ]
pkg> activate .
pkg> instantiate
```

This installs Pun and all of its dependencies using the project manifest shipped with the repository.

### Running a model

The core workflow is built around two primitives:

- `simulate(p)` executes a Pun program and returns `(value, log_weight, basis)`. The `basis`
  is a sparse matrix describing the reference measure for the returned value; reuse it when
  you want to score the same sample later (even under a different distribution).
- `assess(p, value, basis)` evaluates the log density of `value` under the program `p` with
  respect to a Hausdorff measure on a manifold locally specified by `basis`. For discrete outputs use `DiscreteBase`. For smooth manifolds
  embedded in \(\mathbb{R}^n\), pass a matrix whose columns span the tangent space—`LebesgueBase(d)`
  covers full-dimensional densities, while lower-dimensional supports (e.g. for distributions on spheres) use
  skinny matrices. If you obtained the sample via `simulate`, you can pass the
  returned basis directly.

```julia
sample, logw, basis = simulate(flip(0.4))
logpdf = assess(flip(0.4), sample, basis)   # ≈ -logw
```

## Testing

The repository ships with an automated test suite that exercises the interpreter, pattern system,
and numerical linear algebra utilities:

```bash
$ julia --project -e 'using Pkg; Pkg.test()'
```
