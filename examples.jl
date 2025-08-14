example() = @prob begin
    u <<= random()
    pair <<= dirac((u, u))
    u >>= dirac(pair[1])
    return pair
end

circle_example(r) = @prob begin
    theta <<= uniform(0, pi/2)
    point <<= dirac((r*cos(theta), r*sin(theta)))
    theta >>= dirac(atan(point[2]/r, point[1]/r))
    return point
end

uniform(a, b) = @prob begin
    x <<= random()
    y <<= dirac(x * (b - a) + a)
    x >>= dirac((y - a) / (b - a))
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
    y <<= dirac(f(x))
    x >>= dirac(finv(y))
    return y
end

geometric(p) = @prob begin
    b <<= flip(p)
    n <<= b ? dirac(0) : pushforward(geometric(p), x -> x + 1, x -> x - 1)
    b >>= dirac(n == 0)
    return n
end

iid(p, n) = @prob begin
    if n == 0
        y <<= dirac([])
        return y
    else
        xs <<= iid(p, n - 1)
        x <<= p
        y <<= dirac([xs..., x])
        (xs, x) >>= dirac((y[1:end-1], y[end]))
        return y
    end
end


categorical(ws) = @prob begin
    cumsum_ws = cumsum(ws ./ sum(ws)) # can use "=" because the expression does not depend on random variables assigned in this @prob scope.
    u <<= random()
    j <<= dirac(findfirst(cumsum_ws .>= u))
    u >>= uniform(j == 1 ? 0 : cumsum_ws[j-1], cumsum_ws[j])
    return j
end

mapM(f, xs) = @prob begin
    if isempty(xs)
        res <<= dirac([])
        return res
    else
        ys <<= mapM(f, xs[1:end-1])
        y <<= f(xs[end])
        res <<= dirac([ys..., y])
        # (ys, y) >>= dirac((res[1:end-1], res[end]))
        ys >>= dirac(res[1:end-1])
        y  >>= dirac(res[end])
        return res
    end
end



shuffle(xs) = @prob begin
    if isempty(xs)
        l <<= dirac([])
        return l
    else
        fst <<= categorical(ones(length(xs)))
        rst <<= shuffle([xs[1:fst-1]..., xs[fst+1:end]...])
        l <<= dirac([xs[fst], rst...])
        # (fst, rest) <- dirac((findfirst(x -> l[1] == x, xs), l[2:end]))
        rst >>= dirac(l[2:end])
        fst >>= dirac(findfirst(x -> l[1] == x, xs))
        return l
    end
end

# function shuffle(xs)
#     if isempty(xs)
#         return dirac([])
#     end

#     @prob begin
#         fst <<= categorical(ones(length(xs)))
#         rst <<= shuffle([xs[1:fst-1]..., xs[fst+1:end]...])
#         l <<= dirac([xs[fst], rst...])
#         rst >>= dirac(l[2:end])
#         fst >>= dirac(findfirst(x -> l[1] == x, xs))
#         return l
#     end
# end

sorted(p) = @prob begin
    xs <<= p
    ys <<= dirac(sort(xs))
    xs >>= shuffle(ys)
    return ys
end

beta(a, b) = @prob begin
    n = a + b - 1
    xs <<= sorted(iid(random(), n))
    x <<= dirac(xs[a])
    xs >>= @prob begin
        prefix <<= sorted(iid(uniform(0, x), a-1))
        suffix <<= sorted(iid(uniform(x, 1), b-1))
        xs <<= dirac([prefix..., x, suffix...])
        prefix >>= dirac(xs[1:a-1])
        suffix >>= dirac(xs[a+1:end])
        return xs
    end
    return x
end

example2(a, b) = @prob begin
    u <<= beta(a, b)
    x <<= dirac((u, u))
    u >>= dirac(x[2])
    return x
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
        x <<= dirac(xs[j])
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
            x >>= length(y[1])==length(acc) ? dirac(y[2]) : dirac(y[1][length(acc)+1])
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
            (loop1, loop2) <<= dirac([loop[1:j-1], loop[j:end]])
            # Unsample the remainder of the loop by running rejection again.
            (loop, j) >>= dirac(([loop1..., loop2...], length(loop1)+1))
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