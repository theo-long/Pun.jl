struct Random <: Program end

random() = Random()

struct Dirac <: Program
    value
end

dirac(value) = Dirac(value)

struct Normal <: Program
    mean :: Float64
    std  :: Float64
    function Normal(mean, std)
        new(DynamicForwardDiff.value(mean), DynamicForwardDiff.value(std))
    end
end

normal(mean, std) = Normal(mean, std)

struct Gamma <: Program
    shape :: Float64
    scale :: Float64
    function Gamma(shape, scale)
        new(DynamicForwardDiff.value(shape), DynamicForwardDiff.value(scale))
    end
end
gamma(shape, scale) = Gamma(shape, scale)


struct Uniform <: Program
    a :: Float64
    b :: Float64
    function Uniform(a, b)
        new(DynamicForwardDiff.value(a), DynamicForwardDiff.value(b))
    end
end
uniform(a, b) = Uniform(a, b)

struct Bernoulli <: Program
    p :: Float64
    function Bernoulli(p)
        new(DynamicForwardDiff.value(p))
    end
end
bernoulli(p) = Bernoulli(p)

struct Categorical <: Program
    probs :: Vector{Float64}
    function Categorical(probs)
        probs = DynamicForwardDiff.value.(probs)
        new(probs ./ sum(probs))
    end
end
categorical(probs) = Categorical(probs)

struct Beta <: Program
    alpha::Float64
    beta::Float64
    function Beta(alpha, beta)
        new(DynamicForwardDiff.value(alpha), DynamicForwardDiff.value(beta))
    end 
end
beta(alpha, beta) = Beta(alpha, beta)

struct DirichletSymmetric <: Program
    alpha::Float64
    k::Int
    function DirichletSymmetric(alpha::Real, k::Int)
        new(DynamicForwardDiff.value(alpha),k)
    end
end
dirichlet(alpha, k) = DirichletSymmetric(alpha, k)

struct DirichletGeneral <: Program
    alphas::Vector{Float64}
    function DirichletGeneral(alphas)
        new(DynamicForwardDiff.value.(alphas))
    end
end
dirichlet(alphas) = DirichletGeneral(alphas)

struct UniformDiscrete <: Program
    a::Int
    b::Int
end
uniform_discrete(a, b) = UniformDiscrete(a, b)

export random, dirac, normal, gamma, bernoulli, categorical, uniform, dirichlet, beta