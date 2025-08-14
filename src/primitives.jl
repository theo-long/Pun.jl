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

export random, dirac, normal