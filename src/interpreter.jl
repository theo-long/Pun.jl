using DynamicForwardDiff, LinearAlgebra
import Distributions

mutable struct EvalState
    vals :: Dict{Symbol, Any}
    cfg :: DynamicForwardDiff.DiffConfig
    tape :: Vector{DynamicForwardDiff.Dual}
    logweight :: Float64

    function EvalState()
        return new(Dict(), DynamicForwardDiff.DiffConfig(), DynamicForwardDiff.Dual[], 0.0)
    end
end

# Interpreter

# Blocks:

function interpret_program(block::Block, state)
    old_vals, state.vals = state.vals, Dict()
    for cmd in block.commands
        interpret_command(cmd, state)
    end

    @assert !isnothing(block.retvar) "Block must end in a return statement."

    retval = state.vals[block.retvar]
    @assert length(state.vals) == 1 "Variables must be unassigned or returned at end of `@prob` block: $(collect([k for k in keys(state.vals) if k != block.retvar]))"

    state.vals = old_vals
    return retval
end

function uninterpret_program(cmd::Block, state, value)
    old_vals, state.vals = state.vals, Dict(cmd.retvar => value)

    for cmd in cmd.commands[end:-1:1]
        uninterpret_command(cmd, state)
    end
    
    state.vals = old_vals
end

# Commands:
function interpret_command(cmd::Assign, state)
    rhs_block = cmd.f([state.vals[k] for k in cmd.free]...)
    state.vals[cmd.symbol] = interpret_program(rhs_block, state)
end

function interpret_command(cmd::Unassign, state)
    rhs_block = cmd.f([state.vals[k] for k in cmd.free]...)
    uninterpret_program(rhs_block, state, state.vals[cmd.symbol])
    delete!(state.vals, cmd.symbol)
end

function uninterpret_command(cmd::Assign, state)
    interpret_command(Unassign(cmd.symbol, cmd.f, cmd.free), state)
end

function uninterpret_command(cmd::Unassign, state)
    interpret_command(Assign(cmd.symbol, cmd.f, cmd.free), state)
end

# Primitives:
function interpret_program(cmd::Random, state)
    val = Base.rand()
    return DynamicForwardDiff.new_dual(state.cfg, val)
end

function uninterpret_program(cmd::Random, state, value)
    if value < 0 || value > 1
        state.logweight = -Inf
    end
    # @assert value >= 0 && value <= 1 "Random value $value is not in [0, 1]"
    push!(state.tape, value)
end

function interpret_program(cmd::Dirac, state)
    return cmd.value
end

function uninterpret_program(cmd::Dirac, state, value)
    # @assert value == cmd.value "Dirac value $value does not match expected value $(cmd.value)"
    # TODO: if not equal, set logweight to -Inf, maybe raise an exception to be caught by `assess`?
    # Need a multiple-dispatch equality test that uses isapprox for floating point.
    if !isapproximately(value, cmd.value)
        state.logweight = -Inf
    end
end

function interpret_program(cmd::Normal, state)
    val = Base.randn() * cmd.std + cmd.mean
    dual_val = DynamicForwardDiff.new_dual(state.cfg, val)
    state.logweight -= Distributions.logpdf(Distributions.Normal(cmd.mean, cmd.std), val)
    return dual_val
end

function uninterpret_program(cmd::Normal, state, value)
    state.logweight += Distributions.logpdf(Distributions.Normal(cmd.mean, cmd.std), DynamicForwardDiff.value(value))
    push!(state.tape, value)
end


# Jacobian corrections:

function accumulate_partials!(val::Bool, partials)
    nothing
end

function accumulate_partials!(val::Int, partials)
    nothing
end

function accumulate_partials!(val::Float64, partials)
    nothing
end

# Structs:
function accumulate_partials!(val, partials)
    # recursively accumulate for all fields
    for field in fieldnames(typeof(val))
        accumulate_partials!(getfield(val, field), partials)
    end
end

function accumulate_partials!(val::Union{Vector,Tuple}, partials)
    for x in val
        accumulate_partials!(x, partials)
    end
end

function accumulate_partials!(val::DynamicForwardDiff.Dual, partials)
    p = DynamicForwardDiff.partials(val)
    !iszero(p) && push!(partials, p)
end

function compute_jacobian_correction(simulate_result)
    output_partials = []
    accumulate_partials!(simulate_result, output_partials)

    if isempty(output_partials)
        return 1.0
    end

    jacobian = hcat(output_partials...)

    # remove any rows which are all 0s--shouldn't be possible in simulate
    # but can happen in assess.
    # @assert all(!iszero(row) for row in eachrow(jacobian))
    if !all(!iszero(row) for row in eachrow(jacobian))
        @warn "Make sure you intended to assess density w.r.t. this base measure."
        # jacobian = hcat(
        #     (row for row in eachrow(jacobian) if !iszero(row))...
        # )
        # TODO: when did this arise in SMCP3 and why was removing the rows
        # the right thing to do? I think basically you could read something
        # from the input trace and discard it, because you only specified an
        # update to the model trace, not the entire model trace.
        @warn jacobian, size(jacobian)
    end

    xsize, ysize = size(jacobian)
    if xsize !== ysize   
        sqrt(abs(det(jacobian * transpose(jacobian))))
    else
        abs(det(jacobian))
    end
end

function simulate(p::Program)
    state = EvalState()
    v = interpret_program(p, state)
    d = compute_jacobian_correction((v, state.tape))
    return v, log(d) + state.logweight
end

function stock(v::Float64, cfg)
    return DynamicForwardDiff.new_dual(cfg, v)
end
function stock(v::Vector, cfg)
    return [stock(x, cfg) for x in v]
end
function stock(v::Tuple, cfg)
    return tuple((stock(x, cfg) for x in v)...)
end
function stock(v::Bool, cfg)
    return v
end
function stock(v::Int, cfg)
    return v
end
function stock(v::Dict, cfg)
    return Dict(k => stock(v[k], cfg) for k in keys(v))
end
function stock(v, cfg)
    # apply stock to each field
    if ismutable(v)
        # For mutable structs, modify in place
        for field in fieldnames(typeof(v))
            setfield!(v, field, stock(getfield(v, field), cfg))
        end
        return v
    elseif v isa NamedTuple
        # For NamedTuples, construct new one with stock'd field values
        field_names = keys(v)
        field_values = [stock(getfield(v, field), cfg) for field in field_names]
        return NamedTuple{field_names}(field_values)
    else
        # For immutable structs, construct a new instance
        field_values = [stock(getfield(v, field), cfg) for field in fieldnames(typeof(v))]
        return typeof(v)(field_values...)
    end
end

# Note: assessing with stock measure.
function assess(p::Program, v)
    s = EvalState()
    v = stock(v, s.cfg)
    uninterpret_program(p, s, v)
    d = compute_jacobian_correction(((), s.tape))
    return log(d) + s.logweight
end

function radon_nikodym(p::Program, q::Program)
    s = EvalState()
    v = interpret_program(q, s)
    uninterpret_program(p, s, v)
    d = compute_jacobian_correction(((), s.tape))
    return v, log(d) + s.logweight
end

# Disintegrate p with respect to the stock measure, yielding an
# unnormalized posterior on x. Using q(y) as the proposal, generate
# a properly weighted sample for the unnormalized posterior.
function disintegrate(p, q, y)
    s = EvalState()
    y = stock(y, s.cfg)
    x = interpret_program(q(y), s)
    uninterpret_program(p, s, (x, y))
    d = compute_jacobian_correction(((), s.tape))
    return x, log(d) + s.logweight
end


# How to do disintegration? 
#    Given y, x <<= q(y)
#             (x, y) >>= p
#    The "Given y" is kind of like what happens
#    when we interpret `return` backward.
#    So really here, we have x = interpret_program(q(y), s);
#                            uninterpret_program(p, (x, y))
# Another way to think about it:
#    assess on the program (x, y) <<= p; x >>= q(y); return y.
#    But, we want to remember x.
#    

export simulate, assess, radon_nikodym, disintegrate