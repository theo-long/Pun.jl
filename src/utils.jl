"Sample from f(x) for x in xs"
mapM(f, xs) = @prob begin
    if isempty(xs)
        return []
    else
        y <<= f(xs[1])
        ys <<= mapM(f, xs[2:end])
        return [y, ys...]
    end
end

"Sample n iid variables from p"
iid(p, n) = @prob begin
    if n == 0
        return []
    else
        x <<= p
        xs <<= iid(p, n - 1)
        return [x, xs...]
    end
end

"""
Given some probabilistic program, constrain some random draws to have specific values.

Currently only supports constraining the value of variables which appear in the return statement
"""
function constrain(model::Block, constraints::Dict{Symbol,<:Any})
    retval_assignments = [c for c in model.commands if c.symbol == model.retvar]
    symbols = keys(constraints)
    @assert length(retval_assignments) == 1
    retval_symbols = first(retval_assignments).free
    if !all([s in retval_symbols for s in symbols])
        throw(
            ErrorException("Can only constrain variables that appear in the return statement: $([s for s in symbols if !(s in retval_symbols)])")
        )
    end
    new_commands = []
    for command in model.commands
        if typeof(command) == Assign && command.symbol in symbols
            new_command = Assign(command.symbol, () -> dirac(constraints[command.symbol]), command.free)
        else
            new_command = command
        end
        push!(new_commands, new_command)
    end
    return Block(new_commands, model.retvar)
end

export mapM, iid, constrain