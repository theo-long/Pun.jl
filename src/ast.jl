abstract type Command end

struct Assign <: Command
    symbol::Symbol
    f::Function # from values for the free variables to a Program
    free::Vector{Symbol}
end

struct Unassign <: Command
    symbol::Symbol
    f::Function
    free::Vector{Symbol}
end

abstract type Program end

mutable struct Block <: Program
    commands::Vector{Command}
    retvar::Union{Symbol,Nothing}
end