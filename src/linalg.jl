using LinearAlgebra
using SparseArrays
using SuiteSparse

"""
Build a SparseMatrixCSC (m-by-n) from a vector of row-dicts.
Each `rows[i]` is a Dict{Int,Float64} giving the nonzeros in row i 
(keys = column indices).
"""
function dictrows_to_sparse(rows::Vector{Dict{Int,Float64}}, ncols::Int; starting_col::Int = 0)
    I = Int[]; J = Int[]; V = Float64[]
    for (i, d) in enumerate(rows)
        for (j, v) in d
            if j > starting_col
                push!(I, i); push!(J, j - starting_col); push!(V, v)
            end
        end
    end
    m = length(rows); n = ncols - starting_col
    return sparse(I, J, V, m, n)
end

"""
Convert a SparseMatrixCSC (m-by-n) into a vector of Dict row-dicts.
Each Dict maps column index → value for that row.
"""
function sparse_to_dictrows(A::SparseMatrixCSC{<:Real,Int}; starting_col=0)
    rows = [Dict{Int,Float64}() for _ in 1:size(A,1)]
    I, J, V = findnz(A)
    for (i, j, v) in zip(I, J, V)
        rows[i][j+starting_col] = float(v)
    end
    return rows
end


function compute_jacobian_correction_assess(tape_partials, num_inputs; tol=nothing)
    if length(tape_partials) == 0
        return 0.0
    end
    M2p = dictrows_to_sparse(Dict{Int,Float64}[partial.values for partial in tape_partials], num_inputs)
    weight = logpdet_from_tall(M2p; tol=tol)
    return weight
end

function attach_tangents(v, cfg, tangents, current)
    structwalk_map_state(v, current, (leaf, idx) -> begin
        if leaf isa Float64
            dual = DynamicForwardDiff.Dual{Nothing,Float64}(leaf, DynamicForwardDiff.Partials{Float64}(tangents[idx], cfg.n_inputs))
            return dual, idx + 1
        else
            return leaf, idx
        end
    end)
end

function expand_basis_rows(ret, basis)
    rows = sparse_to_dictrows(basis)
    expanded_rows = Dict{Int,Float64}[]
    idx = Ref(1)
    structwalk_each(ret) do leaf
        if leaf isa DynamicForwardDiff.Dual
            @assert idx[] <= length(rows) "Basis rows mismatch"
            push!(expanded_rows, rows[idx[]])
            idx[] += 1
        elseif leaf isa Float64
            push!(expanded_rows, Dict{Int,Float64}())
        end
    end
    @assert idx[] - 1 == length(rows) "Basis rows mismatch: used $(idx[] - 1) but had $(length(rows))"
    return dictrows_to_sparse(expanded_rows, size(basis, 2))
end

function assess(program, value, basis)
    state = EvalState()
    tangents = sparse_to_dictrows(basis)
    k, d = size(basis)
    state.cfg.n_inputs[] = d
    value, current = attach_tangents(value, state.cfg, tangents, 1)
    @assert current == k + 1 "$(current) != $(k + 1)"
    uninterpret_program(program, state, value, Set(1:d))
    partials = []
    accumulate_partials!(state.tape, partials)
    return state.logweight + compute_jacobian_correction_assess(partials, state.cfg.n_inputs[])
end




function logpdet_from_tall(matrix; tol=nothing)
    if size(matrix, 2) == 0
        return 0.0
    end
    F = isnothing(tol) ? qr(matrix) : qr(matrix; tol=tol)
    t = rank(F)
    R = F.R[1:t, 1:t]
    return sum(log, abs.(diag(R)))
end

function make_Vx_vol_normalized(Phat; tol)
    F = isnothing(tol) ? qr(Phat) : qr(Phat; tol=tol)
    p = F.pcol
    r = rank(F)
    R11 = F.R[1:r, 1:r]
    Vraw = Phat[:,p[1:r]]
    d = diag(R11)
    thr = isnothing(tol) ? sqrt(eps(Float64)) : tol
    if any(abs.(d) .<= thr)
        error("R11 has tiny diagonal; adjust QR tolerance.")
    end
    D = spdiagm(0 => 1.0 ./ d)
    return Vraw * D
end

function compute_jacobian_correction_simulate(Phat, Qblk)
    s   = opnorm(Phat, 1)                # cheap scale of A
    rtol = 1e-10                         # choose per problem
    atol = 1e-14
    tol  = max(rtol*s, atol)

    M1 = [Phat; Qblk]
    logJ1 = logpdet_from_tall(M1; tol=tol)
    Vx = make_Vx_vol_normalized(Phat; tol=tol)
    return logJ1, Vx
end


function simulate(p; args=nothing, arg_basis=nothing)

    state = EvalState()

    if !isnothing(args)
        @assert !isnothing(arg_basis) "simulate: if keyword argument `args` is provided, `arg_basis` must also be provided"
        
        tangents = sparse_to_dictrows(arg_basis)
        k, d = size(arg_basis)
        state.cfg.n_inputs[] = d
        args, current = attach_tangents(args, state.cfg, tangents, 1)

        # Assume args is a namedtuple and loop through its named fields
        for field in fieldnames(typeof(args))
            val = getfield(args, field)
            state.vals[field] = val

            # deps should really be computed while attaching tangents
            state.deps[field] = Set()
            accumulate_deps!(val, state.deps[field])
        end
    end

    ret, = interpret_program(p, state)
    ret_partials, tape_partials = [], []
    accumulate_partials!(ret, ret_partials)
    accumulate_partials!(state.tape, tape_partials)
    num_inputs = state.cfg.n_inputs[]
    Phat = dictrows_to_sparse(Dict{Int,Float64}[partial.values for partial in ret_partials], num_inputs)
    Qblk = dictrows_to_sparse(Dict{Int,Float64}[partial.values for partial in tape_partials], num_inputs)

    logpdf, basis = compute_jacobian_correction_simulate(Phat, Qblk)
    basis = expand_basis_rows(ret, basis)
    value = unstock(ret)
    return value, state.logweight + logpdf, basis
end





const DiscreteBase = spzeros(0,0)
LebesgueBase(dim) = spdiagm(ones(dim))

export DiscreteBase, LebesgueBase, simulate, assess
