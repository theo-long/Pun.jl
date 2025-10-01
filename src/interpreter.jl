using DynamicForwardDiff, LinearAlgebra
import Distributions

mutable struct EvalState
    # current values of all Pun variables in scope
    vals :: Dict{Symbol, Any}
    # configuration tracking growing number of continuous sources w.r.t. which
    # we are computing derivatives
    cfg :: DynamicForwardDiff.DiffConfig
    # auxiliary outputs that will make the overall program a bijection
    # from sampled random variables ("source") to outputs (retval and tape)
    tape :: Vector{DynamicForwardDiff.Dual}
    # the L(tape)/K(source) density ratio, up to a change-of-variables 
    # correction that will be added at the end
    logweight :: Float64
    # which continuous source inputs (as numbered in `cfg`) does each Pun variable depend on?
    deps :: Dict{Symbol, Set{Int}}
    # which continuous source inputs does the ambient Julia environment depend on?
    ambient :: Set{Int}
    function EvalState()
        return new(Dict(), DynamicForwardDiff.DiffConfig(), DynamicForwardDiff.Dual[], 0., Dict(), Set())
    end
    function EvalState(vals, cfg, tape, logweight, deps, ambient)
        return new(vals, cfg, tape, logweight, deps, ambient)
    end
end

# Interpreter

# Blocks:

function interpret_program(block::Block, state)
    # old_vals, state.vals = state.vals, Dict()
    for cmd in block.commands
        interpret_command(cmd, state)
    end

    @assert !isnothing(block.retvar) "Block must end in a return statement."

    retval = state.vals[block.retvar]
    @assert length(state.vals) == 1 "Variables must be unassigned or returned at end of `@prob` block: $(collect([k for k in keys(state.vals) if k != block.retvar]))"

    # state.vals = old_vals
    return retval, state.deps[block.retvar]
end

function uninterpret_program(cmd::Block, state, value, deps)
    #old_vals, state.vals = state.vals, Dict(cmd.retvar => value)
    state.vals[cmd.retvar] = value
    state.deps[cmd.retvar] = deps
    for cmd in cmd.commands[end:-1:1]
        uninterpret_command(cmd, state)
    end
    
    #state.vals = old_vals
end

# Commands:
function interpret_command(cmd::Assign, state)
    rhs_block = cmd.f([state.vals[k] for k in cmd.free]...)

    # Set up subscope
    vals, deps = state.vals, state.deps
    state.vals, state.deps = Dict{Symbol, Any}(), Dict{Symbol, Set{Int}}()
    new_ambient_deps = Set()
    for k in cmd.free
        union!(new_ambient_deps, deps[k])
    end
    setdiff!(new_ambient_deps, state.ambient)
    union!(state.ambient, new_ambient_deps)

    # Run command in substate
    vals[cmd.symbol], deps[cmd.symbol] = interpret_program(rhs_block, state)

    # Restore old state
    setdiff!(state.ambient, new_ambient_deps)
    state.vals, state.deps = vals, deps
end

function interpret_command(cmd::Unassign, state)
    rhs_block = cmd.f([state.vals[k] for k in cmd.free]...)

    # Set up subscope
    vals, deps = state.vals, state.deps
    state.vals, state.deps = Dict{Symbol, Any}(), Dict{Symbol, Set{Int}}()
    new_ambient_deps = Set()
    for k in cmd.free
        union!(new_ambient_deps, deps[k])
    end

    # Unrun command in substate
    uninterpret_program(rhs_block, state, vals[cmd.symbol], deps[cmd.symbol])
    delete!(vals, cmd.symbol)
    delete!(deps, cmd.symbol)

    # Restore old state
    setdiff!(new_ambient_deps, state.ambient)
    union!(state.ambient, new_ambient_deps)
    state.vals, state.deps = vals, deps
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
    dual_val = DynamicForwardDiff.new_dual(state.cfg, val)
    return dual_val, Set(state.cfg.n_inputs[])
end

function uninterpret_program(cmd::Random, state, value, deps)
    if value < 0 || value > 1
        state.logweight = -Inf
    end
    # @assert value >= 0 && value <= 1 "Random value $value is not in [0, 1]"
    push!(state.tape, value)
end


function accumulate_deps!(val, deps)
    structwalk_each(val) do leaf
        if leaf isa DynamicForwardDiff.Dual
            p = DynamicForwardDiff.partials(leaf)
            !iszero(p) && union!(deps, keys(p.values))
        end
    end
end

function interpret_program(cmd::Dirac, state)
    deps = Set()
    accumulate_deps!(cmd.value, deps)
    return cmd.value, deps
end

function uninterpret_program(cmd::Dirac, state, value, deps)
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
    return dual_val, Set(state.cfg.n_inputs[])
end

function uninterpret_program(cmd::Normal, state, value, deps)
    state.logweight += Distributions.logpdf(Distributions.Normal(cmd.mean, cmd.std), DynamicForwardDiff.value(value))
    push!(state.tape, value)
end


# Jacobian corrections:

function accumulate_partials!(val, partials)
    structwalk_each(val) do leaf
        if leaf isa DynamicForwardDiff.Dual
            p = DynamicForwardDiff.partials(leaf)
            !iszero(p) && push!(partials, p)
        end
    end
end

function unstock(value)
    structwalk_map(value, leaf -> leaf isa DynamicForwardDiff.Dual ? DynamicForwardDiff.value(leaf) : leaf)
end

function stock(value, cfg)
    structwalk_map(value, leaf -> leaf isa Float64 ? DynamicForwardDiff.new_dual(cfg, leaf) : leaf)
end
