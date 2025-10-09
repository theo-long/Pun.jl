mapM(f, xs) = @prob begin
    if isempty(xs)
        return []
    else
        y <<= f(xs[1])
        ys <<= mapM(f, xs[2:end])
        return [y, ys...]
    end
end

export mapM