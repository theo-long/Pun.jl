import DynamicForwardDiff

# Utility predicates ---------------------------------------------------------

structwalk_is_leaf(x) = x isa Number || x isa AbstractString || x isa Symbol ||
                        x === nothing || x isa Missing || x isa Function ||
                        x isa DynamicForwardDiff.Partials

# Mapping --------------------------------------------------------------------

function structwalk_map(value, f::Function)
    new_value, _ = structwalk_map_state(value, nothing, (leaf, state) -> (f(leaf), state))
    return new_value
end

function structwalk_map_state(value, state, f)
    if structwalk_is_leaf(value)
        return f(value, state)
    elseif value isa Tuple
        items = Vector{Any}(undef, length(value))
        for (i, v) in pairs(value)
            mapped, state = structwalk_map_state(v, state, f)
            items[i] = mapped
        end
        return tuple(items...), state
    elseif value isa NamedTuple
        names = keys(value)
        items = Vector{Any}(undef, length(names))
        for (i, name) in enumerate(names)
            mapped, state = structwalk_map_state(getfield(value, name), state, f)
            items[i] = mapped
        end
        return NamedTuple{names}(Tuple(items)), state
    elseif value isa AbstractArray
        result = similar(value, Any)
        for idx in eachindex(value)
            mapped, state = structwalk_map_state(value[idx], state, f)
            result[idx] = mapped
        end
        return result, state
    elseif value isa AbstractDict
        result = empty(value)
        for (k, v) in pairs(value)
            mapped, state = structwalk_map_state(v, state, f)
            result[k] = mapped
        end
        return result, state
    elseif value isa AbstractSet
        result = empty(value)
        for v in value
            mapped, state = structwalk_map_state(v, state, f)
            push!(result, mapped)
        end
        return result, state
    elseif Base.isstructtype(typeof(value)) && !(value isa Module)
        T = typeof(value)
        fields = fieldnames(T)
        items = Vector{Any}(undef, length(fields))
        for (i, field) in enumerate(fields)
            mapped, state = structwalk_map_state(getfield(value, field), state, f)
            items[i] = mapped
        end
        return T(items...), state
    else
        return f(value, state)
    end
end


# Folding --------------------------------------------------------------------

function structwalk_fold(value, acc, f)
    if structwalk_is_leaf(value)
        return f(acc, value)
    elseif value isa Tuple
        for v in value
            acc = structwalk_fold(v, acc, f)
        end
        return acc
    elseif value isa NamedTuple
        for name in keys(value)
            acc = structwalk_fold(getfield(value, name), acc, f)
        end
        return acc
    elseif value isa AbstractArray
        for v in value
            acc = structwalk_fold(v, acc, f)
        end
        return acc
    elseif value isa AbstractDict
        for v in values(value)
            acc = structwalk_fold(v, acc, f)
        end
        return acc
    elseif value isa AbstractSet
        for v in value
            acc = structwalk_fold(v, acc, f)
        end
        return acc
    elseif Base.isstructtype(typeof(value)) && !(value isa Module)
        for field in fieldnames(typeof(value))
            acc = structwalk_fold(getfield(value, field), acc, f)
        end
        return acc
    else
        return f(acc, value)
    end
end

function structwalk_each(value, f::Function)
    structwalk_fold(value, nothing, (acc, leaf) -> begin
        f(leaf)
        return acc
    end)
end

structwalk_each(f::Function, value) = structwalk_each(value, f)
