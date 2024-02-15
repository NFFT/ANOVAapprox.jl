module ANOVAapprox

using GroupedTransforms,
 LinearAlgebra, IterativeSolvers, LinearMaps, Distributed, SpecialFunctions, Optim, Plots

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

    sign(lval)*sign(rval) == 1 && error("bisection: root is not between l and r")
    if lval > 0
        gun = fun
        fun = t -> -gun(t)
    end

    m = 0.0
    for _ in 1:maxiter
        m = (l+r)/2
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


"""
    C = fitrate(X, y)
fits a function of the form
  ``(C[4] - x)*(C[1] + C[2]*x^C[3])``

# Input
 - `X::Vector{Float64}`
 - `y::Vector{Float64}`
 - `verbose::Bool = false`

# Output
 - `C::Vector{Float64}`: coefficients of the approximation
"""

function fitrate(x, y)
    if sum(y .> 2*y[round(Int, sqrt(length(y)))]) <= 1 && length(y)>25
        return (y[round(Int, length(x)/2)], 0)
    end

    function fitrate_wo(x, y)
        res = optimize(
            C -> maximum(y.*x.^(2C[1]))/minimum(y.*x.^(2C[1])),
            [1.]
        )
        t = Optim.minimizer(res)[1]

        D1 = minimum(y.*x.^(2t))
        D2 = maximum(y.*x.^(2t))
        return (D1, D2, t)
    end

    ts = zeros(Float64, length(x))
    for idx in 1:length(x)
        D1, D2, t = fitrate_wo(x[1:idx], y[1:idx])
        ts[idx] = t
    end

    function find_plateau(x, y; n = 1, Δ = (maximum(y)-minimum(y))/100)
        h = zeros(length(x))
        idcs_prev = zeros(Int, length(x))
        idcs_next = zeros(Int, length(x))

        for idx in 1:length(ts)
            tmp = ( abs.(ts[idx] .- ts) .> Δ/2 )

            idx_prev = findprev(tmp, idx)
            idx_prev = ( isnothing(idx_prev) ? idx : idx_prev+1 )

            idx_next = findnext(tmp, idx)
            idx_next = ( isnothing(idx_next) ? idx : idx_next-1 )

            h[idx] = abs.(x[idx_prev:idx_next-1]-x[idx_prev+1:idx_next]) |> sum
            idcs_prev[idx] = idx_prev
            idcs_next[idx] = idx_next
        end
        h[h .== 0] .= -Inf

        idcs = zeros(Int, n)
        for i in 1:n
            idcs[i] = argmax(h)
            h[idcs_prev[idcs[i]]:idcs_next[idcs[i]]] .= -Inf
            if all(h .== -Inf)
                idcs = idcs[1:i]
                break
            end
        end

        return (idcs_prev[idcs], idcs_next[idcs])
    end

    n = 3
    idcs_prev, idcs_next = find_plateau(log.(x), ts; n = n)
    n = length(idcs_prev)
    

    offset = zeros(n)

    for i in 1:n
        idx_prev = idcs_prev[i]
        idx_next = idcs_next[i]

        D1, D2, t = fitrate_wo(x[1:idx_next], y[1:idx_next])
        if idx_next > 1
            offset[i] = sum( i -> log(x[i+1]/x[i])*(log(sqrt(D1*D2)*x[i]^(-2t)) - log(y[i]) )^2, 1:idx_next-1)/log(x[idx_next])
        end
    end

# compare with least squares on all data
    w = log.(x[2:end]./x[1:end-1])
    tmp = lsqr(diagm(sqrt.(w))*[ones(length(x)-1) log.(x[1:end-1])], sqrt.(w).*log.(y[1:end-1]))
    D = exp(tmp[1])
    t = tmp[2]/-2
    offset_lsqr = sum( i -> log(x[i+1]/x[i])*(log(D*x[i]^(-2t)) .- log(y[i]) )^2, 1:length(x)-1)/log(x[end])

    if 2*offset_lsqr < minimum(offset)
        return (y[round(Int, length(x)/2)], 0)
    else
        idx = idcs_next[argmin(offset)]
        D1, D2, t = fitrate_wo(x[1:idx], y[1:idx])
        return (sqrt(D1*D2), t)
    end
end

function testrate(S::Vector{Vector{Float64}},C::Vector{Vector{Float64}},t::Float64)::Vector{Bool}
    E = [((C[i][4]).-(1:length(S[i]))).*(C[i][1].+C[i][2].*(1:length(S[i])).^(C[i][3])) for i=1:length(C)]
    return [sum(abs.(E[i].-S[i]))/length(S[i])<t for i=1:length(C)]
end


include("fista.jl")
include("approx.jl")
include("errors.jl")
include("analysis.jl")

end # module
