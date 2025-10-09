using LogExpFunctions

# struct Observation
#     # f
#     y
#     basis
# end


function importance_sampling(p, q, y, basis, n)
    particles = []
    weights = []
    for i in 1:n
        _, w, _ = simulate((@prob y -> begin
                x <<= q(y)

                # a hack to save `x` even though we are unassigning it
                nothing .<<= push!(particles, unstock(x))

                (x, y) >>= p
                return nothing
            end); args=(; y), arg_basis=basis)
        push!(weights, w)
    end

    return collect(zip(particles, weights))
end

function importance_resampling(p, q, y, basis, n)
    weighted_particles = importance_sampling(p, q, y, basis, n)
    particles, log_weights = getindex.(weighted_particles, 1), getindex.(weighted_particles, 2)
    weights = exp.(log_weights .- logsumexp(log_weights))
    resampler = @prob begin
        index <<= categorical(weights)
        return index
    end
    index, _, _ = simulate(resampler)
    return particles[index]
end


# External version of importance sampling
# function importance_sampling(p, q, y, basis, n)
#     particles = []
#     dim = size(basis, 2)
#     tangents = sparse_to_dictrows(basis)

#     for i in 1:n
#         state = EvalState()
#         state.cfg.n_inputs[] = dim
#         y_, = attach_tangents(y, state.cfg, tangents, 1)
#         val, deps = interpret_program(q(y_), state)
#         uninterpret_program(p, state, (val, y_), union(deps, Set(1:dim)))
#         tape_partials = DynamicForwardDiff.Partials[]
#         accumulate_partials!(state.tape, tape_partials)
#         Qblk = dictrows_to_sparse(Dict{Int,Float64}[p.values for p in tape_partials], state.cfg.n_inputs[])
#         correction = logpdet_from_tall(Qblk; tol=1e-14)
#         push!(particles, (unstock(val), state.logweight + correction))
#     end

#     return particles
# end

function mh(p, q, y, w, basis)
    (new_y, new_w, new_basis), alpha = propose_mh(p, q, y, w, basis)
    if log(rand()) < alpha
        return (new_y, new_w, new_basis)
    else
        return (y, w, basis)
    end
end

function propose_mh(p, q, y, w, basis)
    new_y, q_ratio, new_basis = simulate((@prob old -> begin
            new <<= q(old)
            old >>= q(new)
            return new
        end); args=(; old=y), arg_basis=basis)
    new_w = assess(p, new_y, new_basis)
    return (new_y, new_w, new_basis), new_w - w + q_ratio
end


# function propose_mh(p, q, y, basis, w)
#     state = EvalState()
#     dim = size(basis, 2)
#     state.cfg.n_inputs[] = dim
#     tangents = sparse_to_dictrows(basis)
#     y_, = attach_tangents(y, state.cfg, tangents, 1)
#     y_new_, deps = interpret_program(q(y_), state)
#     state.ambient = deps   # probably unnecessary
#     uninterpret_program(q(y_new_), state, y_, Set(1:dim))
#     y_new_partials = DynamicForwardDiff.Partials[]
#     tape_partials = DynamicForwardDiff.Partials[]
#     accumulate_partials!(y_new_, y_new_partials)
#     accumulate_partials!(state.tape, tape_partials)
#     Phat = dictrows_to_sparse(Dict{Int,Float64}[p.values for p in y_new_partials], state.cfg.n_inputs[])
#     Qblk = dictrows_to_sparse(Dict{Int,Float64}[p.values for p in tape_partials], state.cfg.n_inputs[])
#     correction, y_new_basis = compute_jacobian_correction_simulate(Phat, Qblk)
#     log_q_ratio = state.logweight + correction # d(H\otimes q)/d(swap*(H \otimes q))(y_new, y)
#     y_new = unstock(y_new_)
#     new_w = assess(p, y_new, y_new_basis)
#     alpha = new_w - w + log_q_ratio
#     return (y_new, y_new_basis, new_w), alpha
# end






# struct IS <: Program
#     p :: Program # over a space X
#     q :: Program # program over X
#     f # X -> Y along which to disintegrate
#     n :: Int
# end


# function interpret_program(is::IS, state)


# end
