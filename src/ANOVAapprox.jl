module ANOVAapprox

using GroupedTransforms,
    LinearAlgebra, IterativeSolvers, LinearMaps, Distributed, SpecialFunctions

bases = ["per", "cos", "cheb", "std", "chui1", "chui2", "chui3", "chui4", "expcos"]
types = Dict(
    "per" => ComplexF64,
    "cos" => Float64,
    "cheb" => Float64,
    "std" => Float64,
    "chui1" => Float64,
    "chui2" => Float64,
    "chui3" => Float64,
    "chui4" => Float64,
    "expcos" => ComplexF64,
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
    "expcos" => Vector{ComplexF64},
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
    "expcos" => "expcos",
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

include("fista.jl")
include("approx.jl")
include("errors.jl")
include("analysis.jl")

end # module
