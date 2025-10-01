"""
    isapproximately(a, b; kwargs...)

Check if two values are approximately equal, handling various types recursively.

For floating-point numbers, uses `isapprox` with the given keyword arguments.
For tuples, structs, namedtuples, vectors, and dicts, recursively compares elements.
For other types (bools, ints, etc.), uses exact equality.

# Arguments
- `a`, `b`: Values to compare
- `kwargs...`: Keyword arguments passed to `isapprox` for floating-point comparisons

# Examples
```julia
isapproximately(1.0, 1.0001)  # true (using isapprox)
isapproximately((1.0, 2.0), (1.0001, 2.0))  # true
isapproximately([1.0, 2.0], [1.0001, 2.0])  # true
isapproximately(Dict(:x => 1.0), Dict(:x => 1.0001))  # true
isapproximately(true, true)  # true (exact equality)
isapproximately(1, 1)  # true (exact equality)
```
"""
function isapproximately(a, b; kwargs...)
    if a isa DynamicForwardDiff.Dual
        a = a.value
    end
    if b isa DynamicForwardDiff.Dual
        b = b.value
    end
    if a isa Number && b isa Number
        if a isa AbstractFloat || b isa AbstractFloat
            return isapprox(a, b; kwargs...)
        else
            return a == b
        end
    elseif (a isa Bool || a isa Integer || a isa String || a isa Symbol) && typeof(a) == typeof(b)
        return a == b
    elseif a isa Tuple && b isa Tuple
        return length(a) == length(b) && all(isapproximately.(a, b; kwargs...))
    elseif a isa NamedTuple && b isa NamedTuple
        return keys(a) == keys(b) && all(isapproximately.(values(a), values(b); kwargs...))
    elseif a isa AbstractVector && b isa AbstractVector
        return length(a) == length(b) && all(isapproximately.(a, b; kwargs...))
    elseif a isa AbstractDict && b isa AbstractDict
        return keys(a) == keys(b) && all(isapproximately.(values(a), values(b); kwargs...))
    elseif a isa AbstractArray && b isa AbstractArray
        return size(a) == size(b) && all(isapproximately.(a, b; kwargs...))
    elseif a isa AbstractSet && b isa AbstractSet
        if length(a) != length(b)
            return false
        end
        
        # Convert to arrays and find matching pairs using approximate equality
        a_array = collect(a)
        b_array = collect(b)
        
        # For each element in a, find a matching element in b
        used_indices = falses(length(b_array))
        for elem_a in a_array
            found_match = false
            for (i, elem_b) in enumerate(b_array)
                if !used_indices[i] && isapproximately(elem_a, elem_b; kwargs...)
                    used_indices[i] = true
                    found_match = true
                    break
                end
            end
            if !found_match
                return false
            end
        end
        return true
    elseif isstructtype(typeof(a)) && isstructtype(typeof(b))
        # Handle structs by comparing all fields
        if fieldnames(typeof(a)) == fieldnames(typeof(b))
            return all(field -> isapproximately(getfield(a, field), getfield(b, field); kwargs...), fieldnames(typeof(a)))
        else
            return false
        end
    else
        # Fallback to exact equality for unknown types
        return a == b
    end
end

# Helper function to check if a type is a struct type
function isstructtype(T::Type)
    return T isa DataType && !isabstracttype(T) && !isprimitivetype(T)
end


