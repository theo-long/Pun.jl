using MacroTools

# Macro implementation
macro prob(ex)
    return parse_prob_block(__source__, ex)
end

format_line_suffix(line::Nothing) = ""
function format_line_suffix(line::LineNumberNode)
    file = line.file
    file_str = file === nothing ? "" : String(file)
    if isempty(file_str)
        return " (at line $(line.line))"
    else
        return " (at $(file_str):$(line.line))"
    end
end

function return_missing_error(missing::AbstractSet{Symbol}, line::Union{Nothing,LineNumberNode})
    names = join(sort!(collect(missing)), ", ")
    noun = length(missing) == 1 ? "random variable" : "random variables"
    pronoun = length(missing) == 1 ? "it" : "them"
    msg = "@prob return must include $noun still in scope: [" * names * "]"
    msg *= ". Unassign $pronoun with `>>=` or include $pronoun in the return value."
    msg *= format_line_suffix(line)
    return msg
end

function throw_pun_error(msg::String, line::Union{Nothing,LineNumberNode})
    if line isa LineNumberNode
        file = line.file === nothing ? "" : String(line.file)
        throw(LoadError(file, line.line, ErrorException(msg)))
    else
        throw(ErrorException(msg))
    end
end

function return_not_pun_error(var::Symbol, line::Union{Nothing,LineNumberNode})
    msg = "@prob return expects a Pun variable, but `$var` has not been introduced via `<<=` or pattern destructuring"
    msg *= format_line_suffix(line)
    return msg
end

const PUN_INFIX_OPS = (Symbol("<<="), Symbol(">>="), Symbol(".<<="), Symbol(".>>="))

function is_pun_call(ex::Expr)
    ex.head in PUN_INFIX_OPS && return true
    return ex.head == :call && !isempty(ex.args) && ex.args[1] in PUN_INFIX_OPS
end
is_pun_call(::Any) = false

function find_illegal_usage(ex, assigned_vars)
    if ex isa Symbol
        return ex in assigned_vars ? ex : nothing
    elseif ex isa Expr
        if ex.head == :return || ex.head == :(=) || is_pun_call(ex)
            return nothing
        end
        for arg in ex.args
            sym = find_illegal_usage(arg, assigned_vars)
            sym !== nothing && return sym
        end
    end
    return nothing
end

function illegal_usage_error(sym::Symbol, line::Union{Nothing,LineNumberNode})
    msg = "Pun variable `$sym` cannot appear outside of an assignment (<<=), unassignment (>>=), or return."
    msg *= format_line_suffix(line)
    return msg
end

function parse_prob_block(source::LineNumberNode, ex)
    body = if ex isa Expr && ex.head == :block
        collect(ex.args)
    else
        Any[ex]
    end

    statements = Any[]
    assigned_vars = Set{Symbol}()
    block_var = gensym(:block)
    ret_seen = Ref(false)
    current_line = Ref{Union{Nothing,LineNumberNode}}(nothing)
    current_file = Ref(source.file)

    for stmt in body
        if stmt isa LineNumberNode
            current_line[] = stmt
            current_file[] = stmt.file
            push!(statements, stmt)
            continue
        end
        if stmt isa Expr && stmt.head != :(=) && stmt.head != :return && !is_pun_call(stmt)
            sym = find_illegal_usage(stmt, assigned_vars)
            if sym !== nothing
                throw_pun_error(illegal_usage_error(sym, current_line[]), current_line[])
            end
        end
        parsed_stmt = parse_expression(stmt, assigned_vars, block_var, ret_seen, current_line[], current_file[])
        push!(statements, parsed_stmt)
    end

    if isempty(statements)
        throw_pun_error("@prob block must contain a return value", current_line[])
    end

    if !ret_seen[]
        msg = "@prob block must explicitly return a value; add `return ...` to the block"
        msg *= format_line_suffix(current_line[])
        throw_pun_error(msg, current_line[])
    end

    let_body = Expr(:block, statements...)
    return esc(:(let $block_var = $(Block)([], nothing); $let_body; $block_var end))
end

function parse_expression(ex, assigned_vars, block_var, ret_seen, current_line, current_file)
    if !(ex isa Expr)
        return ex
    end

    # Handle assignment: x <<= expr
    if @capture(ex, lhs_ <<= rhs_)
        return parse_assignment(lhs, rhs, assigned_vars, block_var)
    # Handle unassignment: x >>= expr  
    elseif @capture(ex, lhs_ >>= rhs_)
        return parse_unassignment(lhs, rhs, assigned_vars, block_var)
    elseif @capture(ex, lhs_ .<<= rhs_)
        return parse_assignment(lhs, :(dirac($rhs)), assigned_vars, block_var)
    elseif @capture(ex, lhs_ .>>= rhs_)
        return parse_unassignment(lhs, :(dirac($rhs)), assigned_vars, block_var)
    # Handle return: return x
    elseif @capture(ex, return var_)
        ret_seen[] = true
        return parse_return(var, assigned_vars, block_var, current_line)
    elseif ex.head == :(=)
        lhs = ex.args[1]
        if lhs isa Symbol && lhs in assigned_vars
            msg = "Found `$(lhs) = ...` inside @prob; use `$(lhs) <<= ...` or `$(lhs) .<<= ...` instead"
            msg *= format_line_suffix(current_line)
            throw_pun_error(msg, current_line)
        end
        new_args = [parse_expression(arg, assigned_vars, block_var, ret_seen, current_line, current_file) for arg in ex.args]
        return Expr(ex.head, new_args...)
    else
        new_args = [parse_expression(arg, assigned_vars, block_var, ret_seen, current_line, current_file) for arg in ex.args]
        return Expr(ex.head, new_args...)
    end
end


function process_pattern!(pattern, var_name, current_expr, d)
    # Given a pattern (e.g. (:x, :y)), we want to return a dictionary mapping 
    # variables to function expressions that would assign them.
    if pattern isa Symbol
        d[pattern] = :($var_name -> $(Dirac)($current_expr))
    elseif pattern isa Expr && pattern.head == :tuple && !is_namedtuple_pattern(pattern)
        for (i, arg) in enumerate(pattern.args)
            process_pattern!(arg, var_name, :($(current_expr)[$(i)]), d)
        end
    elseif is_dict_pattern(pattern)
        process_dict_pattern!(pattern, var_name, current_expr, d)
    elseif is_struct_pattern(pattern)
        process_struct_pattern!(pattern, var_name, current_expr, d)
    elseif is_vector_pattern(pattern)
        process_vector_pattern!(pattern, var_name, current_expr, d)
    elseif is_namedtuple_pattern(pattern)
        process_namedtuple_pattern!(pattern, var_name, current_expr, d)
    else
        dump(pattern)
        @error "Unsupported pattern: $pattern"
    end
end

# Helper function to detect if a pattern is a struct pattern
function is_struct_pattern(pattern)
    if !(pattern isa Expr && pattern.head == :call && pattern.args[1] isa Symbol)
        return false
    end
    
    # Check if it's a known non-struct expression
    head_symbol = pattern.args[1]
    if head_symbol in [:tuple, :getfield, :getindex, :+, :-, :*, :/, :^]
        return false
    end
    
    # Check if it looks like a comparison operator
    if string(head_symbol) in ["<", ">", "<=", ">=", "==", "!="]
        return false
    end
    
    # Must have at least one field pattern
    return length(pattern.args) > 1
end

# Helper function to detect if a pattern is a Dictionary pattern
function is_dict_pattern(pattern)
    return pattern isa Expr && 
           pattern.head == :call && 
           pattern.args[1] == :Dict &&
           length(pattern.args) > 1
end

# Helper function to detect if a pattern is a NamedTuple pattern
function is_namedtuple_pattern(pattern)
    return pattern isa Expr && 
           pattern.head == :tuple &&
           all(arg -> arg isa Expr && arg.head == :(=), pattern.args)
end

# Helper function to detect if a pattern is a Vector pattern
function is_vector_pattern(pattern)
    return pattern isa Expr && pattern.head == :vect
end

# Helper function to process Vector patterns
function process_vector_pattern!(pattern, var_name, current_expr, d)
    # pattern is of the form: [pattern1, pattern2, pattern3, ...]
    patterns = pattern.args
    
    # Check for multiple splatted patterns
    splatted_patterns = [p for p in patterns if p isa Expr && p.head == :(...)]
    if length(splatted_patterns) > 1
        error("Multiple splatted patterns in vector pattern are not supported: $pattern. Only one pattern can be splatted.")
    end
    
    # Find the splatted pattern and its position
    splatted_idx = findfirst(p -> p isa Expr && p.head == :(...), patterns)
    
    if splatted_idx === nothing
        # No splatting: simple vector pattern
        for (i, sub_pattern) in enumerate(patterns)
            element_access = :($(current_expr)[$i])
            process_pattern!(sub_pattern, var_name, element_access, d)
        end
    else
        # Handle splatting with unified approach
        splatted_pattern = patterns[splatted_idx]
        inner_pattern = splatted_pattern.args[1]  # Extract the pattern inside ...
        
        # Calculate the range for the splatted pattern
        if splatted_idx == 1
            # Splat at beginning: [rest..., x, y]
            # rest gets elements 1:end-2, x gets element end-1, y gets element end
            remaining_elements = patterns[2:end]
            
            # Process splatted pattern - use a range that will be evaluated at runtime
            element_access = :($(current_expr)[1:end-$(length(remaining_elements))])
            process_pattern!(inner_pattern, var_name, element_access, d)
            
            # Process remaining elements
            for (i, sub_pattern) in enumerate(remaining_elements)
                element_access = :($(current_expr)[end-$(length(remaining_elements)-i)])
                process_pattern!(sub_pattern, var_name, element_access, d)
            end
            
        elseif splatted_idx == length(patterns)
            # Splat at end: [x, y, rest...]
            # x gets element 1, y gets element 2, rest gets elements 3:end
            first_elements = patterns[1:end-1]
            
            # Process first elements
            for (i, sub_pattern) in enumerate(first_elements)
                element_access = :($(current_expr)[$i])
                process_pattern!(sub_pattern, var_name, element_access, d)
            end
            
            # Process splatted pattern
            element_access = :($(current_expr)[$(length(first_elements)+1):end])
            process_pattern!(inner_pattern, var_name, element_access, d)
            
        else
            # Splat in middle: [x, rest..., y, z]
            # x gets element 1, rest gets elements 2:end-2, y gets element end-1, z gets element end
            before_splat = patterns[1:splatted_idx-1]
            after_splat = patterns[splatted_idx+1:end]
            
            # Process elements before splat
            for (i, sub_pattern) in enumerate(before_splat)
                element_access = :($(current_expr)[$i])
                process_pattern!(sub_pattern, var_name, element_access, d)
            end
            
            # Process splatted pattern
            element_access = :($(current_expr)[$(length(before_splat)+1):end-$(length(after_splat))])
            process_pattern!(inner_pattern, var_name, element_access, d)
            
            # Process elements after splat
            for (i, sub_pattern) in enumerate(after_splat)
                element_access = :($(current_expr)[end-$(length(after_splat)-i)])
                process_pattern!(sub_pattern, var_name, element_access, d)
            end
        end
    end
end

# Helper function to process struct patterns
function process_struct_pattern!(pattern, var_name, current_expr, d)
    # pattern is of the form: StructName(field1, field2, ...)
    field_patterns = pattern.args[2:end]

    for (i, field_pattern) in enumerate(field_patterns)
        field_access = :(getfield($current_expr, $i))
        process_pattern!(field_pattern, var_name, field_access, d)
    end
end

# Helper function to process Dictionary patterns
function process_dict_pattern!(pattern, var_name, current_expr, d)
    # pattern is of the form: Dict(:key => pattern, :otherkey => other_pattern, ...)
    key_value_pairs = pattern.args[2:end]
    for pair in key_value_pairs
        if pair isa Expr && pair.head == :call && pair.args[1] == :(=>)
            key = pair.args[2]
            value_pattern = pair.args[3]
            # Access dictionary value using the key
            value_access = :($(current_expr)[$key])
            process_pattern!(value_pattern, var_name, value_access, d)
        else
            error("Unsupported entry in Dict pattern: $pair")
        end
    end
end

# Helper function to process NamedTuple patterns
function process_namedtuple_pattern!(pattern, var_name, current_expr, d)
    # pattern is of the form: (key=pattern, key2=pattern2, ...)
    for pair in pattern.args
        if pair isa Expr && pair.head == :(=)
            key = pair.args[1]
            value_pattern = pair.args[2]
            # Access NamedTuple field using the key
            value_access = :(getfield($current_expr, $(QuoteNode(key))))
            process_pattern!(value_pattern, var_name, value_access, d)
        end
    end
end

function handle_pattern(pattern, var_name, unpack, block_var)
    # We assume that there is a variable var_name that we are unpacking into pattern.
    # If `unpack` is true, we want:
    #    pattern <- var_name
    # i.e.
    #    (to be preceeded by var_name <- e)
    #    x <- dirac(var_name[...])
    #    y <- dirac(var_name[...])
    #    var_name -> dirac(pattern)
    # If `unpack` is false, we want:
    #    pattern -> var_name
    # i.e.
    #    var_name <- dirac(pattern)
    #    x -> dirac(var_name[...])
    #    y -> dirac(var_name[...])
    #    (to be followed by var_name -> e)
    
    d = Dict()
    process_pattern!(pattern, var_name, var_name, d)

    fwd_func = unpack ? Assign : Unassign
    bwd_func = unpack ? Unassign : Assign

    stmts = [:(push!($block_var.commands, $fwd_func($(QuoteNode(variable)), $func_expr, [$(QuoteNode(var_name))]))) for (variable, func_expr) in d]
    vars = collect(keys(d))
    vars_expr = Expr(:tuple, vars...)
    push!(stmts, :(push!($block_var.commands, $bwd_func($(QuoteNode(var_name)), $vars_expr -> $(Dirac)($pattern), $vars))))

    if unpack
        return stmts, vars
    else
        return reverse(stmts), vars
    end
end


function parse_assignment(lhs, rhs, assigned_vars, block_var)

    # Find free variables in RHS that were previously assigned in this block
    free_vars = find_free_variables_in_block(rhs, assigned_vars)
    
    # Create function that captures free variables
    if isempty(free_vars)
        func_expr = :(() -> $rhs)
    else
        # Create function that takes free variables as parameters
        func_expr = create_capturing_function(rhs, free_vars)
    end

    lhs_name = lhs isa Symbol ? lhs : gensym("temp")
    stmt = :(push!($block_var.commands, $(Assign)($(QuoteNode(lhs_name)), $func_expr, $(free_vars))))

    if lhs isa Symbol
        push!(assigned_vars, lhs)
        return stmt
    end

    if !(lhs isa Expr)
        error("Left-hand side of assignment must be a pattern, got: $lhs")
    end

    # Check if it's a valid pattern (tuple or struct)
    if !(lhs.head == :tuple || is_struct_pattern(lhs) || is_dict_pattern(lhs) || is_namedtuple_pattern(lhs) || is_vector_pattern(lhs))
        error("Left-hand side of assignment must be a valid pattern (tuple, struct, Dict, NamedTuple, or Vector), got: $lhs")
    end

    stmts, vars = handle_pattern(lhs, lhs_name, true, block_var)
    push!(assigned_vars, vars...)

    return Expr(:block, stmt, stmts...)
end

function parse_unassignment(lhs, rhs, assigned_vars, block_var)
    lhs_name = lhs isa Symbol ? lhs : gensym("temp")
    
    if lhs isa Symbol 
        delete!(assigned_vars, lhs)
    else
        # Check if it's a valid pattern (tuple or struct)
        if !(lhs.head == :tuple || is_struct_pattern(lhs) || is_dict_pattern(lhs) || is_namedtuple_pattern(lhs) || is_vector_pattern(lhs))
            error("Left-hand side of unassignment must be a valid pattern (tuple, struct, Dict, NamedTuple, or Vector), got: $lhs")
        end
        
        stmts, vars = handle_pattern(lhs, lhs_name, false, block_var)
        for var in vars
            delete!(assigned_vars, var)
        end
    end
    
    # Find free variables in RHS that were previously assigned in this block
    free_vars = find_free_variables_in_block(rhs, assigned_vars)
    
    # Create function that captures free variables
    if isempty(free_vars)
        func_expr = :(() -> $rhs)
    else
        # Create function that takes free variables as parameters
        func_expr = create_capturing_function(rhs, free_vars)
    end
    
    stmt = :(push!($block_var.commands, $(Unassign)($(QuoteNode(lhs_name)), $func_expr, $(free_vars))))

    return (lhs isa Symbol ? stmt : Expr(:block, stmts..., stmt))
end

function parse_return(var, assigned_vars, block_var, current_line)
    if var isa Symbol
        if !(var in assigned_vars)
            throw_pun_error(return_not_pun_error(var, current_line), current_line)
        end
        missing = setdiff(assigned_vars, Set([var]))
        if !isempty(missing)
            throw_pun_error(return_missing_error(missing, current_line), current_line)
        end
        delete!(assigned_vars, var)
        return :($block_var.retvar = $(QuoteNode(var)))
    end

    tmpvar = gensym("retval")
    stmts, vars = handle_pattern(var, tmpvar, false, block_var)
    missing_vars = Symbol[]
    for v in vars
        if !(v in assigned_vars)
            push!(missing_vars, v)
        end
    end
    if !isempty(missing_vars)
        names = join(sort!(missing_vars), ", ")
        msg = "@prob return pattern refers to variables not in scope: [" * names * "]"
        msg *= format_line_suffix(current_line)
        throw_pun_error(msg, current_line)
    end
    missing = setdiff(assigned_vars, Set(vars))
    if !isempty(missing)
        throw_pun_error(return_missing_error(missing, current_line), current_line)
    end
    for v in vars
        delete!(assigned_vars, v)
    end
    return Expr(:block, stmts..., :($block_var.retvar = $(QuoteNode(tmpvar))))
end


# Track free variables among bindings created earlier in the same `@prob` block,
# taking care to respect lexical scopes such as anonymous functions, `let`, and
# `for` bindings so we do not capture shadowed names.
function find_free_variables_in_block(ex, assigned_vars)
    assigned = Set(assigned_vars)
    used_vars = Symbol[]
    find_variables!(ex, used_vars, Set{Symbol}(), assigned)
    return used_vars
end

function find_variables!(ex, vars, bound, assigned)
    if ex isa Symbol
        if ex in assigned && !(ex in bound) && !(ex in vars) && !(ex in [:+, :-, :*, :/, :^])
            push!(vars, ex)
        end
    elseif ex isa Expr
        if ex.head == :-> && length(ex.args) == 2
            new_bound = union(bound, collect_bound_symbols(ex.args[1]))
            find_variables!(ex.args[2], vars, new_bound, assigned)
        elseif ex.head == :function && length(ex.args) >= 2
            params = collect_function_params(ex.args[1])
            new_bound = union(bound, params)
            for arg in ex.args[2:end]
                find_variables!(arg, vars, new_bound, assigned)
            end
        elseif ex.head == :let && length(ex.args) >= 1
            nargs = length(ex.args)
            new_bound = copy(bound)
            for binding in ex.args[1:nargs-1]
                if binding isa Expr && binding.head == :(=) && length(binding.args) == 2
                    find_variables!(binding.args[2], vars, new_bound, assigned)
                    union!(new_bound, collect_bound_symbols(binding.args[1]))
                else
                    find_variables!(binding, vars, new_bound, assigned)
                end
            end
            find_variables!(ex.args[end], vars, new_bound, assigned)
        elseif ex.head == :for && length(ex.args) >= 2
            new_bound = union(bound, collect_for_bound_symbols(ex.args[1:end-1]))
            find_variables!(ex.args[end], vars, new_bound, assigned)
        else
            for arg in ex.args
                find_variables!(arg, vars, bound, assigned)
            end
        end
    end
end

function collect_bound_symbols(ex)
    symbols = Set{Symbol}()
    collect_bound_symbols!(symbols, ex)
    return symbols
end

function collect_bound_symbols!(symbols, ex)
    if ex isa Symbol
        push!(symbols, ex)
    elseif ex isa Expr
        if ex.head == :(::) || ex.head == :(=)
            collect_bound_symbols!(symbols, ex.args[1])
        elseif ex.head == :tuple || ex.head == :parameters
            for arg in ex.args
                collect_bound_symbols!(symbols, arg)
            end
        elseif ex.head == :... && length(ex.args) == 1
            collect_bound_symbols!(symbols, ex.args[1])
        else
            for arg in ex.args
                collect_bound_symbols!(symbols, arg)
            end
        end
    end
end

function collect_function_params(ex)
    params = Set{Symbol}()
    if ex isa Symbol
        return params
    elseif ex isa Expr && ex.head == :call
        for arg in ex.args[2:end]
            collect_bound_symbols!(params, arg)
        end
        return params
    else
        collect_bound_symbols!(params, ex)
        return params
    end
end

function collect_for_bound_symbols(generators)
    symbols = Set{Symbol}()
    for generator in generators
        if generator isa Expr && generator.head == :generator
            collect_for_bound_symbols!(symbols, generator.args)
        else
            collect_for_bound_symbols!(symbols, generator)
        end
    end
    return symbols
end

function collect_for_bound_symbols!(symbols, ex)
    if ex isa Expr && ex.head == :(=)
        collect_bound_symbols!(symbols, ex.args[1])
    elseif ex isa Expr
        for arg in ex.args
            collect_for_bound_symbols!(symbols, arg)
        end
    end
end

function create_capturing_function(body, free_vars)
    # Create a function that takes the free variables as parameters
    if isempty(free_vars)
        return :(() -> $body)
    else
        # Create function that takes free variables as parameters
        # Don't substitute variable names - let them be resolved in the local scope
        param_expr = Expr(:tuple, free_vars...)
        return :($param_expr -> $body)
    end
end

export @prob
