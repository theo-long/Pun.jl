using Test
using SparseArrays
using Pun
using Pun: simulate, assess, LebesgueBase, DiscreteBase
using Distributions

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

iid(p, n) = @prob begin
    if n == 0
        return []
    else
        x <<= p
        xs <<= iid(p, n - 1)
        return [x, xs...]
    end
end

categorical(ws) = @prob begin
    W = sum(ws)
    if W == 0
        j <<= dirac(NaN)
        return j
    else
        cumsum_ws = cumsum(ws ./ W)
        u <<= random()
        j .<<= findfirst(cumsum_ws .>= u)
        u >>= uniform(j == 1 ? 0.0 : cumsum_ws[j-1], cumsum_ws[j])
        return j
    end
end

shuffle(xs) = @prob begin
    if isempty(xs)
        return []
    else
        j <<= categorical(ones(length(xs)))
        fst .<<= xs[j]
        rst <<= shuffle(vcat(xs[1:j-1], xs[j+1:end]))
        j >>= categorical(xs .== fst)
        return [fst, rst...]
    end
end

sorted_uniforms(n) = @prob begin
    xs <<= iid(random(), n)
    ys .<<= sort(xs)
    xs >>= shuffle(ys)
    return ys
end

paired_example() = @prob begin
    u <<= random()
    return (u, u)
end

uniform_on_circle(r) = @prob begin
    theta <<= uniform(-pi, pi)
    point <<= dirac((r*cos(theta), r*sin(theta)))
    theta >>= dirac(atan(point[2]/r, point[1]/r))
    return point
end

rayleigh() = @prob begin
    x <<= normal(0, 1)
    y <<= normal(0, 1)
    r <<= dirac(sqrt(x^2 + y^2))
    (x, y) >>= uniform_on_circle(r)
    return r
end

linear_pushforward() = @prob begin
    u <<= random()
    x <<= dirac(3u + 1)
    u >>= dirac((x - 1) / 3)
    return x
end

sorted(p) = @prob begin
    xs <<= p
    ys .<<= sort(xs)
    xs >>= shuffle(ys)
    return ys
end

⊗(p, q) = @prob begin
    x <<= p
    y <<= q
    return (x, y)
end

beta(a, b) = @prob begin
    n = a + b - 1
    xs <<= sorted(iid(random(), n))
    x .<<= xs[a]
    xs >>= @prob begin
        (prefix, suffix) <<= sorted(iid(uniform(0, x), a-1)) ⊗ sorted(iid(uniform(x, 1), b-1))
        xs .<<= [prefix..., x, suffix...]
        (prefix, suffix) .>>= (xs[1:a-1], xs[a+1:end])
        return xs
    end
    return x
end

@testset "Pun.jl" begin
    @testset "simulate vs assess" begin
        normal_prog = @prob begin
            x <<= normal(0.0, 1.0)
            return x
        end
        value, logwt, basis = simulate(normal_prog)
        @test size(basis) == (1, 1)
        @test assess(normal_prog, value, basis) ≈ -logwt atol=1e-9
        @test logwt isa Float64

        value, logwt, basis = simulate(flip(0.3))
        expected = value ? log(0.3) : log1p(-0.3)
        @test basis == DiscreteBase
        @test assess(flip(0.3), value, basis) ≈ expected atol=1e-9
        @test logwt ≈ -expected atol=1e-9
    end

    @testset "Dirac validation" begin
        deterministic = @prob begin
            x <<= dirac(1.0)
            return x
        end
        val_sim, logwt_sim, basis_sim = simulate(deterministic)
        @test val_sim == 1.0
        @test logwt_sim == 0.0
        @test size(basis_sim) == (1, 0)
        @test basis_sim == spzeros(1, 0)

        basis = spzeros(1, 0)
        @test assess(deterministic, 1.0, basis) == 0.0
        @test assess(deterministic, 2.0, basis) == -Inf
    end

    @testset "Pattern destructuring" begin
        struct Point
            x::Float64
            y::Float64
        end

        pattern_prog = @prob begin
            (a, b) <<= dirac((1.0, -2.0))
            Point(cx, cy) <<= dirac(Point(3.0, 4.5))
            return (a, b, cx, cy)
        end

        val, weight, basis = simulate(pattern_prog)
        @test val == (1.0, -2.0, 3.0, 4.5)
        @test weight ≈ 0.0 atol=1e-12
        @test size(basis) == (4, 0)
        @test basis == spzeros(4, 0)
    end

    @testset "Macro hygiene" begin
        prog = @prob begin
            x <<= dirac(1.0)
            y <<= (x -> x)(dirac(3.0))
            return (x, y)
        end
        @test prog.commands[2] isa Pun.Assign
        @test isempty(prog.commands[2].free)
    end

    @testset "Explicit return required" begin
        block = :(begin
            x <<= dirac(1.0)
            x
        end)
        @test_throws LoadError Pun.parse_prob_block(LineNumberNode(0, Symbol("test")), block)
    end

    @testset "Return must cover scope" begin
        block = :(begin
            x <<= dirac(1.0)
            y <<= dirac(2.0)
            return x
        end)
        err = try
            Pun.parse_prob_block(LineNumberNode(0, Symbol("test")), block)
            nothing
        catch e
            e
        end
        @test err isa Exception
        @test occursin("y", sprint(showerror, err))
    end

    @testset "Return of non-Pun variables" begin
        block = :(begin
            x = dirac(1.0)
            return x
        end)
        err = try
            Pun.parse_prob_block(LineNumberNode(0, Symbol("test")), block)
            nothing
        catch e
            e
        end
        @test err isa Exception
        @test occursin("has not been introduced", sprint(showerror, err))
    end

    @testset "Using Pun variables outside assignments" begin
        block = :(begin
            b <<= flip(0.5)
            if b
                return true
            else
                return false
            end
        end)
        err = try
            Pun.parse_prob_block(LineNumberNode(0, Symbol("test")), block)
            nothing
        catch e
            e
        end
        @test err isa Exception
        @test occursin("b", sprint(showerror, err))
    end

    @testset "Arrow parameters" begin
        prog = @prob (x, y) -> begin
            y >>= dirac(x)
            return x
        end
        @test prog isa Pun.Block

        @test_throws LoadError eval(:(@prob (x, y) -> begin
            return x
        end))
    end

    @testset "isapproximately" begin
        struct Box
            center::Tuple{Float64, Float64}
            tags::Dict{Symbol,Float64}
        end

        a = Box((1.0, 2.0), Dict(:x => 1.0, :y => 2.0))
        b = Box((1.0 + 1e-8, 2.0 - 1e-8), Dict(:x => 1.0 + 1e-8, :y => 2.0 - 1e-8))
        @test Pun.isapproximately(a, b; atol=1e-6)

        set_a = Set([(1.0, 2.0), (3.0, 4.0)])
        set_b = Set([(1.0 + 1e-8, 2.0 - 1e-8), (3.0, 4.0)])
        @test Pun.isapproximately(set_a, set_b; atol=1e-6)
    end

    @testset "Sparse conversions" begin
        rows = [Dict(1 => 1.0, 3 => -2.0), Dict(2 => 3.5)]
        A = Pun.dictrows_to_sparse(rows, 4)
        @test size(A) == (2, 4)
        reconstructed = Pun.sparse_to_dictrows(A)
        @test reconstructed == rows

        tall = spdiagm(0 => [2.0, 3.0])
        @test Pun.logpdet_from_tall(tall) ≈ log(6.0) atol=1e-12
    end

    @testset "Examples-inspired" begin
        @testset "uniform" begin
            a, b = 0.2, 0.6
            value, logwt, basis = simulate(uniform(a, b))
            @test a <= value <= b
            @test size(basis) == (1, 1)
            @test assess(uniform(a, b), value, basis) ≈ -logwt atol=1e-9
            expected = logpdf(Distributions.Uniform(a, b), value)
            @test assess(uniform(a, b), value, basis) ≈ expected atol=1e-9
        end

        @testset "sorted iid" begin
            n = 4
            value, logwt, basis = simulate(sorted_uniforms(n))
            @test length(value) == n
            @test issorted(collect(value))
            @test size(basis) == (n, n)
            @test assess(sorted_uniforms(n), value, basis) ≈ -logwt atol=1e-9
        end

        @testset "paired tuple" begin
            value, logwt, basis = simulate(paired_example())
            @test value[1] == value[2]
            @test size(basis) == (2, 1)
            @test assess(paired_example(), value, basis) ≈ -logwt atol=1e-9
        end

        @testset "linear pushforward" begin
            value, logwt, basis = simulate(linear_pushforward())
            @test 1 <= value <= 4
            @test size(basis) == (1, 1)
            @test assess(linear_pushforward(), value, basis) ≈ -logwt atol=1e-9
            expected = logpdf(Distributions.Uniform(1, 4), value)
            @test assess(linear_pushforward(), value, basis) ≈ expected atol=1e-9
        end

        @testset "rayleigh" begin
            value, logwt, basis = simulate(rayleigh())
            @test value >= 0
            @test size(basis) == (1, 1)
            @test assess(rayleigh(), value, basis) ≈ -logwt atol=1e-9
            expected = logpdf(Distributions.Rayleigh(), value)
            @test assess(rayleigh(), value, basis) ≈ expected atol=1e-9
        end

        @testset "beta" begin
            a, b = 2, 3
            value, logwt, basis = simulate(beta(a, b))
            @test 0.0 <= value <= 1.0
            @test size(basis) == (1, 1)
            expected = logpdf(Distributions.Beta(a, b), value)
            @test assess(beta(a, b), value, basis) ≈ expected atol=1e-9
        end

        @testset "categorical" begin
            ws = [0.2, 0.3, 0.5]
            value, logwt, basis = simulate(categorical(ws))
            @test value in 1:length(ws)
            @test basis == DiscreteBase
            expected = log(ws[value])
            @test assess(categorical(ws), value, basis) ≈ expected atol=1e-9
        end

        @testset "shuffle" begin
            xs = 1:3
            value, logwt, basis = simulate(shuffle(collect(xs)))
            @test sort(value) == collect(xs)
            @test basis == DiscreteBase
            expected = -log(factorial(length(xs)))
            @test assess(shuffle(collect(xs)), value, basis) ≈ expected atol=1e-9
        end
    end
end
