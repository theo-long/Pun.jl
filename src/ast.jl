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

function Base.show(io::IO, p::Block)
    for command in p.commands
        println("\t", command)
    end
    print("return", p.retvar)
end

function Base.show(io::IO, c::Assign)
    println("Assign ", c.symbol, " := ", c.f)
    print("\tFree ", c.free)
end

function Base.show(io::IO, c::Unassign)
    println("Unassign ", c.symbol, " := ", c.f)
    print("\tFree ", c.free)
end