using MacroTools

# Macro implementation
macro prob(ex)
    return parse_prob_block(ex)
end

function parse_prob_block(ex)
    # Convert to expression if it's a block
    if @capture(ex, begin; body__; end)
        # Start with empty block
        statements = []
        assigned_vars = Set{Symbol}()  # Track variables assigned in this block
        block_var = gensym(:block)  # Use gensym to avoid conflicts in nested blocks
        
        # Parse each statement and add to block
        for stmt in body
            parsed_stmt = parse_expression(stmt, assigned_vars, block_var)
            push!(statements, parsed_stmt)
        end
        
        # Build the let expression with proper scoping
        let_body = Expr(:block, statements...)
        # Escape the entire output so all variables resolve in caller's scope
        return esc(:(let $block_var = $(Block)([], nothing); $let_body; $block_var end))
    else
        error("Expected a begin...end block after `@prob`")
    end
end

function parse_expression(ex, assigned_vars, block_var)
    if !(ex isa Expr)
        return ex
    end

    # Handle assignment: x <<= expr
    if @capture(ex, lhs_ <<= rhs_)
        return parse_assignment(lhs, rhs, assigned_vars, block_var)
    # Handle unassignment: x >>= expr  
    elseif @capture(ex, lhs_ >>= rhs_)
        return parse_unassignment(lhs, rhs, assigned_vars, block_var)
    # Handle return: return x
    elseif @capture(ex, return var_)
        return parse_return(var, assigned_vars, block_var)
    else
        new_args = [parse_expression(arg, assigned_vars, block_var) for arg in ex.args]
        return Expr(ex.head, new_args...)
    end
end


function process_pattern!(pattern, var_name, current_expr, d)
    # Given a pattern (e.g. (:x, :y)), we want to return a dictionary mapping 
    # variables to function expressions that would assign them.
    if pattern isa Symbol
        d[pattern] = :($var_name -> $(Dirac)($current_expr))
    elseif pattern isa Expr && pattern.head == :tuple
        for (i, arg) in enumerate(pattern.args)
            process_pattern!(arg, var_name, :($(current_expr)[$(i)]), d)
        end
    else
        @error "Unsupported pattern: $pattern"
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

    stmts, vars = handle_pattern(lhs, lhs_name, true, block_var)
    push!(assigned_vars, vars...)

    return Expr(:block, stmt, stmts...)
end

function parse_unassignment(lhs, rhs, assigned_vars, block_var)
    lhs_name = lhs isa Symbol ? lhs : gensym("temp")
    
    if lhs isa Symbol 
        delete!(assigned_vars, lhs)
    else
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

function parse_return(var, assigned_vars, block_var)
    if var isa Symbol
        return :($block_var.retvar = $(QuoteNode(var)))
    end

    tmpvar = gensym("retval")
    stmts, vars = handle_pattern(var, tmpvar, false, block_var)
    for v in vars
        delete!(assigned_vars, v)
    end
    return Expr(:block, stmts..., :($block_var.retvar = $(QuoteNode(tmpvar))))
end


# TODO: The below code was written by Cursor, and it's not quite correct:
# it doesn't correctly handle shadowing of variables. For example, if `x` is
# an assigned variable, then `y <<= (x -> x)(dirac(3))` should not be treated
# as having `x` as a free variable, but this code will treat it as such.
# That said, it doesn't appear to cause problems to be overly conservative here.
function find_free_variables_in_block(ex, assigned_vars)
    # Find all variables used in the expression
    used_vars = Symbol[]
    find_variables!(ex, used_vars)
    
    # Return only those that were previously assigned in this block, with no duplicates
    # Function parameters and other local variables should not be treated as free variables
    # that need capturing - they should be available in the local scope
    return unique([var for var in used_vars if var in assigned_vars])
end

function find_variables!(ex, vars)
    if ex isa Symbol
        # Don't include operators or special symbols
        if !(ex in [:+, :-, :*, :/, :^])
            push!(vars, ex)
        end
    elseif ex isa Expr
        # Other expressions - check all arguments
        for arg in ex.args
            find_variables!(arg, vars)
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