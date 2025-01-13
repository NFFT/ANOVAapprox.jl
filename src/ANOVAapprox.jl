module ANOVAapprox

using GroupedTransforms
using LinearAlgebra
using IterativeSolvers
using LinearMaps
using SpecialFunctions
using Statistics
using MultivariateAnomalies
using Base.Threads

bases = ["per", "cos", "cheb", "std", "chui1", "chui2", "chui3", "chui4", "mixed"]
types = Dict(
    "per" => ComplexF64,
    "cos" => Float64,
    "cheb" => Float64,
    "std" => Float64,
    "chui1" => Float64,
    "chui2" => Float64,
    "chui3" => Float64,
    "chui4" => Float64,
    "mixed" => ComplexF64,
)
vtypes = Dict(
    "per" => Vector{ComplexF64},
    "cos" => Vector{Float64},
    "cheb" => Vector{Float64},
    "std" => Vector{Float64},
    "chui1" => Vector{Float64},
    "chui2" => Vector{Float64},
    "chui3" => Vector{Float64},
    "chui4" => Vector{Float64},
    "mixed" => Vector{ComplexF64},
)
gt_systems = Dict(
    "per" => "exp",
    "cos" => "cos",
    "cheb" => "cos",
    "std" => "cos",
    "chui1" => "chui1",
    "chui2" => "chui2",
    "chui3" => "chui3",
    "chui4" => "chui4",
    "mixed" => "mixed",
)

function get_orderDependentBW(U::Vector{Vector{Int}}, N::Vector{Int})::Vector{Int}
    N_bw = zeros(Int64, length(U))

    for i = 1:length(U)
        if U[i] == []
            N_bw[i] = 0
        else
            N_bw[i] = N[length(U[i])]
        end
    end

    return N_bw
end

function bisection(l, r, fun; maxiter = 1_000)
    lval = fun(l)
    rval = fun(r)

    sign(lval) * sign(rval) == 1 && error("bisection: root is not between l and r")
    if lval > 0
        gun = fun
        fun = t -> -gun(t)
    end

    m = 0.0
    for _ = 1:maxiter
        m = (l + r) / 2
        mval = fun(m)
        abs(mval) < 1e-16 && break
        if mval < 0
            l = m
            lval = mval
        else
            r = m
            rval = mval
        end
    end
    return m
end

include("fista.jl")
include("approx.jl")
include("errors.jl")
include("analysis.jl")
include("trafo.jl")

end # module
