# struct Observation
#     # f
#     y
#     basis
# end


# External version of importance sampling
function importance_sampling(p, q, y, basis, n)
    particles = []
    dim = size(basis, 2)
    tangents = sparse_to_dictrows(basis)

    for i in 1:n
        state = EvalState()
        state.cfg.n_inputs[] = dim
        y_, = attach_tangents(y, state.cfg, tangents, 1)
        val, deps = interpret_program(q(y_), state)
        uninterpret_program(p, state, (val, y_), union(deps, Set(1:dim)))
        tape_partials = DynamicForwardDiff.Partials[]
        accumulate_partials!(state.tape, tape_partials)
        Qblk = dictrows_to_sparse(Dict{Int,Float64}[p.values for p in tape_partials], state.cfg.n_inputs[])
        correction = logpdet_from_tall(Qblk; tol=1e-14)
        push!(particles, (unstock(val), state.logweight + correction))
    end

    return particles
end





# struct IS <: Program
#     p :: Program # over a space X
#     q :: Program # program over X
#     f # X -> Y along which to disintegrate
#     n :: Int
# end


# function interpret_program(is::IS, state)
    

# end