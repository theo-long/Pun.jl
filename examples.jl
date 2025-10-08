f() = @prob begin
    x <<= random()
    y <<= dirac(x * 2) # equivalent to y <<= dirac(x * 2)
    x >>= dirac(y / 2) # equivalent to x >>= dirac(y / 2)
    return y
end

uniform(a, b) = @prob begin
    x <<= random()
    y .<<= x * (b - a) + a
    x .>>= (y - a) / (b - a)
    return y
end

flip(p) = @prob begin
    u <<= random()
    b <<= dirac(u < p)
    u >>= b ? uniform(0, p) : uniform(p, 1)
    return b
end

pushforward(p, f, finv) = @prob begin
    x <<= p
    y .<<= f(x)
    x .>>= finv(y)
    return y
end

geometric(p) = @prob begin
    b <<= flip(p)
    n <<= if b
        dirac(0)
    else
        pushforward(geometric(p), x -> x + 1, x -> x - 1)
    end
    b .>>= n == 0
    return n
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
        cumsum_ws = cumsum(ws ./ W) # can use "=" because the expression does not depend on random variables assigned in this @prob scope.
        u <<= random()
        j .<<= findfirst(cumsum_ws .>= u)
        u >>= uniform(j == 1 ? 0 : cumsum_ws[j-1], cumsum_ws[j])
        return j
    end
end

mapM(f, xs) = @prob begin
    if isempty(xs)
        return []
    else
        y <<= f(xs[1])
        ys <<= mapM(f, xs[2:end])
        return [y, ys...]
    end
end

shuffle(xs) = @prob begin
    if isempty(xs)
        return []
    else
        # Select a random index to move to the front.
        j <<= categorical(ones(length(xs)))
        fst .<<= xs[j]
        # Shuffle the remainder of the list.
        rst <<= shuffle(vcat(xs[1:j-1], xs[j+1:end]))
        # Uncompute the chosen index, by choosing among the indices equal to `fst`
        j >>= categorical(xs .== fst)
        # Return the shuffled list.
        return [fst, rst...]
    end
end

# Exact if p is exchangeable.
# Valid [unbiased] if p(permute(xs)) > 0 whenever p(xs) > 0.
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

⨵(p, q) = @prob begin
    x <<= p
    y <<= q(x)
    return (x, y)
end

⨴(p, q) = @prob begin
    y <<= q
    x <<= p(y)
    return (x, y)
end

beta(a, b) = @prob begin
    n = a + b - 1
    xs <<= sorted(iid(random(), n))
    x .<<= xs[a]
    xs >>= @prob begin
        (prefix, suffix) <<= sorted(iid(uniform(0, x), a - 1)) ⊗ sorted(iid(uniform(x, 1), b - 1))
        xs .<<= [prefix..., x, suffix...]
        (prefix, suffix) .>>= (xs[1:a-1], xs[a+1:end])
        return xs
    end
    return x
end


example() = @prob begin
    u <<= random()
    return (u, u)
end

circle_example(r) = @prob begin
    theta <<= uniform(0, pi / 2)
    point <<= dirac((r * cos(theta), r * sin(theta)))
    theta >>= dirac(atan(point[2] / r, point[1] / r))
    return point
end


uniform_on_circle(r) = @prob begin
    theta <<= uniform(-pi, pi)
    point <<= dirac((r * cos(theta), r * sin(theta)))
    theta >>= dirac(atan(point[2] / r, point[1] / r))
    return point
end

example2(a, b) = @prob begin
    u <<= beta(a, b)
    return (u, u)
end

rayleigh() = @prob begin
    x <<= normal(0, 1)
    y <<= normal(0, 1)
    r <<= dirac(sqrt(x^2 + y^2))
    (x, y) >>= uniform_on_circle(r)
    return r
end

circle_example_2(r) = @prob begin
    u1 <<= normal(0, 1)
    u2 <<= normal(0, 1)
    z .<<= sqrt(u1^2 + u2^2)
    (x, y) .<<= (u1 * r / z, u2 * r / z)
    (u1, u2) .>>= (x * z / r, y * z / r)
    z >>= rayleigh()
    return (x, y)
end

absnormal(std) = @prob begin
    z <<= normal(0, std)
    x .<<= abs(z)
    z >>= @prob begin
        is_pos <<= flip(0.5)
        z .<<= is_pos ? x : -x
        is_pos .>>= z > 0
        return z
    end
    return x
end

betabernExact(a, b) = @prob begin
    u <<= beta(a, b)
    y <<= flip(u)
    u >>= beta(a + (y ? 0 : 1), b + (y ? 1 : 0))
    return y
end


betabernIS(a, b) = @prob begin
    u <<= beta(a, b)
    y <<= flip(u)
    u >>= @prob begin
        xs <<= iid(beta(a, b), 10)
        j <<= let ws = [y ? p : 1 - p for p in xs]
            categorical(ws)
        end
        x .<<= xs[j]
        xs >>= mapM(i -> i == j ? dirac(x) : beta(a, b), collect(1:10))
        j >>= categorical(ones(10))
        return x
    end
    return y
end

# Generate samples from p until one is accepted by f;
# return all samples in a list.
rejection_traced(p, f) = @prob begin
    x <<= p
    xs <<= f(x) ? dirac([]) : rejection_traced(p, f)
    return [x, xs...]
end

# Generate one sample from p conditioned on f
rejection(p, f) = @prob begin
    [rejections..., accepted] <<= rejection_traced(p, f)

    # Clean up rejections
    rejections >>= @prob begin
        # Choose a random prefix of a fresh rejection loop.
        loop <<= rejection_traced(p, f)
        j <<= categorical(ones(length(loop)))
        rejections .<<= loop[1:j-1]

        # Clean up auxiliary randomness
        loop >>= @prob begin
            suffix <<= rejection_traced(p, f)
            loop .<<= [rejections..., suffix...]
            suffix .>>= loop[j:end]
            return loop
        end
        j .>>= length(rejections) + 1

        return rejections
    end
    return accepted
end

bivgauss() = @prob begin
    x <<= normal(0, 1)
    y <<= normal(0, 1)
    (z, w) <<= dirac((2x + y, y - 3))
    (x, y) >>= dirac(let myY = w + 3
        ((z - myY) / 2, myY)
    end)
    return (z, w)
end

exponential(rate) = @prob begin
    u <<= uniform(0, 1)
    x .<<= -log(1 - u) / rate
    u >>= uniform(0, 1 - exp(-rate * x))
    return x
end

gamma(shape::Int, scale) = @prob begin
    if shape < 1
        error("shape must be >= 1")
    end
    z <<= exponential(1 / scale)
    x <<= if shape > 1
        @prob begin
            u <<= gamma(shape - 1, scale)
            x .<<= u + z
            u .>>= x - z
            return x
        end
    else
        dirac(z)
    end
    z .>>= shape > 1 ? 1 / scale : x
    scaled_x .<<= scale * x
    x .>>= scaled_x / scale
    return scaled_x
end


# # something like this may be more efficient than
# # the manual iid implementation above.
# # but maybe list comprehension or looping syntax in 
# # the core language could help?
# struct IID <: Program
#     p :: Program
#     n :: Int
# end

# function interpret_program(p::IID, s::EvalState)
#     return [interpret_program(p.p, s) for i in 1:p.n]
# end
# function uninterpret_program(p::IID, s::EvalState, v)
#     for x in v
#         uninterpret_program(p.p, s, x)
#     end
# end


# Loops: a variable can be assigned so long as it has been unassigned.
# xs .<<= []
# for n in 1:4
#     x  <<= normal(0, 1)
#     xs <<= push(&xs, &x)
# end
# return xs

# A nicer way to express the following which is already valid (I think)
# xs .<<= []
# for n in 1:4
#    x <<= normal(0, 1)
#    ys <<= [xs..., x]
#    xs .>>= ys[1:end-1]
#    x .>>= ys[end]
#    xs .<<= ys
#    ys .>>= xs
# end

# But maybe we can even have mutable implementations, where:
#   once &xs has been given to us, we own that storage?
#   (there is an issue where ys .<<= xs does not copy xs...)
#   (and therefore mutating could cause ys to change, which is bad)
#   (need to think more through "reversible mutation" -- the work by Charles
#    re pointers and data structures in Tower may be relevant)
#    Still, we could use persistent data structures so that push is log-time.

