f() = @prob begin
    x <<= random()
    y .<<= x * 2 # equivalent to y <<= dirac(x * 2)
    x .>>= y / 2 # equivalent to x >>= dirac(y / 2)
    return y
end

uniform(a, b) = @prob begin
    x  <<= random()
    y .<<= x * (b - a) + a 
    x .>>= (y - a) / (b - a)
    return y
end

flip(p) = @prob begin
    u  <<= random()
    b .<<= u < p
    u  >>= b ? uniform(0, p) : uniform(p, 1)
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
    n <<= b ? dirac(0) : pushforward(geometric(p), x -> x + 1, x -> x - 1)
    b .>>= n == 0
    return n
end

iid(p, n) = @prob begin
    if n == 0
        return []
    else
        x  <<= p
        xs <<= iid(p, n - 1)
        return [x, xs...]
    end
end

categorical(ws) = @prob begin
    cumsum_ws = cumsum(ws ./ sum(ws)) # can use "=" because the expression does not depend on random variables assigned in this @prob scope.
    u <<= random()
    j .<<= findfirst(cumsum_ws .>= u)
    u >>= uniform(j == 1 ? 0 : cumsum_ws[j-1], cumsum_ws[j])
    return j
end

mapM(f, xs) = @prob begin
    if isempty(xs)
        return []
    else
        y  <<= f(xs[1])
        ys <<= mapM(f, xs[2:end])
        return [y, ys...]
    end
end

shuffle(xs) = @prob begin
    if isempty(xs)
        l .<<= []
        return l
    else
        # Select a random index to move to the front.
        j <<= categorical(ones(length(xs)))
        fst .<<= xs[j]
        # Shuffle the remainder of the list.
        rst <<= shuffle([xs[1:j-1]..., xs[j+1:end]...])
        # Uncompute the chosen index, by choosing among the indices equal to `fst`
        j >>= categorical(xs .== fst)
        # Return the shuffled list.
        return [fst, rst...]
    end
end

sorted(p) = @prob begin
    xs <<= p
    ys .<<= sort(xs)
    xs >>= shuffle(ys)
    return ys
end

beta(a, b) = @prob begin
    n = a + b - 1
    xs <<= sorted(iid(random(), n))
    x .<<= xs[a]
    xs >>= @prob begin
        prefix <<= sorted(iid(uniform(0, x), a-1))
        suffix <<= sorted(iid(uniform(x, 1), b-1))
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
    theta <<= uniform(0, pi/2)
    point <<= dirac((r*cos(theta), r*sin(theta)))
    theta >>= dirac(atan(point[2]/r, point[1]/r))
    return point
end

example2(a, b) = @prob begin
    u <<= beta(a, b)
    return (u, u)
end


betabernExact(a, b) = @prob begin
    u <<= beta(a, b)
    y <<= flip(u)
    u >>= beta(a + (y ? 0 : 1), b + (y ? 1 : 0))
    return y
end


# 
betabernIS(a, b) = @prob begin
    u <<= beta(a, b)
    y <<= flip(u)
    u >>= @prob begin
        xs <<= iid(beta(a, b), 10)
        j <<= let ws = [y ? p : 1 - p for p in xs]; categorical(ws) end
        x .<<= xs[j]
        xs >>= mapM(i -> i == j ? dirac(x) : beta(a, b), collect(1:10))
        j >>= categorical(ones(10))
        return x
    end
    return y
end

function rejection(p, f)
    function rejection_traced(acc)
        @prob begin
            x <<= p
            y <<= if f(x)
                dirac((acc, x))
            else
                rejection_traced([acc..., x])
            end
            x .>>= length(y[1])==length(acc) ? y[2] : y[1][length(acc)+1]
            return y
        end
    end

    @prob begin
        # Run rejection sampling to get a list of rejections (loop) and an accepted sample (y)
        (loop, y) <<= rejection_traced([])

        # To unsample the loop:
        loop >>= @prob begin
            # Run rejection again
            (loop, z) <<= rejection_traced([])
            # Choose a random prefix of `loop` to return
            j <<= categorical(ones(length(loop)+1))
            (loop1, loop2) .<<= [loop[1:j-1], loop[j:end]]
            # Unsample the remainder of the loop by running rejection again.
            (loop, j) .>>= ([loop1..., loop2...], length(loop1)+1)
            (loop2, z) >>= rejection_traced([])
            return loop1
        end

        # Return the accepted sample.
        return y
    end
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


