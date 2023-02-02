@doc raw"""
    approx

A struct to hold the scattered data function approximation.

# Fields
* `basis::String` - basis of the function space; currently choice of `"per"` (exponential functions), `"cos"` (cosine functions), `"cheb"` (Chebyshev basis),`"std"`(transformed exponential functions), `"chui1"` (Haar wavelets), `"chui2"` (Chui-Wang wavelets of order 2),`"chui3"`  (Chui-Wang wavelets of order 3) ,`"chui4"` (Chui-Wang wavelets of order 4)
* `X::Matrix{Float64}` - scattered data nodes with d rows and M columns
* `y::Union{Vector{ComplexF64},Vector{Float64}}` - M function values (complex for `basis = "per"`, real ortherwise)
* `U::Vector{Vector{Int}}` - a vector containing susbets of coordinate indices
* `N::Vector{Int}` - bandwdiths for each ANOVA term
* `trafo::GroupedTransform` - holds the grouped transformation
* `fc::Dict{Float64,GroupedCoefficients}` - holds the GroupedCoefficients after approximation for every different regularization parameters

# Constructor
    approx( X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, U::Vector{Vector{Int}}, N::Vector{Int}, basis::String = "cos" )

# Additional Constructor
    approx( X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, ds::Int, N::Vector{Int}, basis::String = "cos" )
"""
mutable struct approx
    basis::String
    X::Matrix{Float64}
    y::Union{Vector{ComplexF64},Vector{Float64}}
    U::Vector{Vector{Int}}
    N::Vector{Int}
    trafo::GroupedTransform
    fc::Dict{Float64,GroupedCoefficients}
    dcos::Vector{Bool}

    function approx(
        X::Matrix{Float64},
        y::Union{Vector{ComplexF64},Vector{Float64}},
        U::Vector{Vector{Int}},
        N::Vector{Int},
        basis::String = "cos",
        dcos::Vector{Bool} = Vector{Bool}([])
    )
        if basis in bases
            M = size(X, 2)
            ds = maximum([length(u) for u in U])

            if !isa(y, vtypes[basis])
                error(
                    "Periodic functions require complex vectors, nonperiodic functions real vectors.",
                )
            end

            if length(y) != M
                error("y needs as many entries as X has columns.")
            end

            if (length(N) != length(U)) && (length(N) != ds)
                error("N needs to have |U| or max |u| entries.")
            end

            if length(N) == ds
                bw = get_orderDependentBW(U, N)
            else
                bw = N
            end

            if basis == "expcos"
                if length(dcos) == 0
                    error("please call approx with dcos for a NFFCT transform.")
                end
                if length(dcos) < maximum(U)[1]
                    error("dcos must have an entry for every dimension.")
                end
            end

            if (
                basis == "per" ||
                basis == "chui1" ||
                basis == "chui2" ||
                basis == "chui3" ||
                basis == "chui4"
            ) && ((minimum(X) < -0.5) || (maximum(X) >= 0.5))
                error("Nodes need to be between -0.5 and 0.5.")
            elseif (basis == "cos") && ((minimum(X) < 0) || (maximum(X) > 1))
                error("Nodes need to be between 0 and 1.")
            elseif (basis == "cheb") && ((minimum(X) < -1) || (maximum(X) > 1))
                error("Nodes need to be between -1 and 1.")
            elseif (basis == "expcos") 
                if sum(dcos) > 0
                    if (minimum(X[dcos,:]) < 0) || (maximum(X[dcos,:]) > 1)
                        error("Nodes must be between 0 and 0.5 for cosinus dimensions.")
                    end
                end
                if sum(.!dcos)>0
                    if (minimum(X[(.!dcos),:]) < -0.5) || (maximum(X[(.!dcos),:]) > 0.5)
                        error("Nodes must be between -0.5 and 0.5 for exponentional dimensions.")
                    end
                end
            end

            Xt = copy(X)

            if basis == "cos"
                Xt ./= 2
            elseif basis == "cheb"
                Xt = acos.(Xt)
                Xt ./= 2 * pi
            elseif basis == "std"
                Xt ./= sqrt(2)
                Xt = erf.(Xt)
                Xt .+= 1
                Xt ./= 4
            end

            trafo = GroupedTransform(gt_systems[basis], U, bw, Xt, dcos)
            return new(basis, X, y, U, bw, trafo, Dict{Float64,GroupedCoefficients}(), dcos)
        else
            error("Basis not found.")
        end
    end
end

function approx(
    X::Matrix{Float64},
    y::Union{Vector{ComplexF64},Vector{Float64}},
    ds::Int,
    N::Vector{Int},
    basis::String = "cos",
    dcos::Vector{Bool} = Vector{Bool}([]),
)
    Uds = get_superposition_set(size(X, 1), ds)
    return approx(X, y, Uds, N, basis, dcos)
end


@doc raw"""
    approximate( a::approx, λ::Float64; max_iter::Int = 50, weights::Union{Vector{Float64},Nothing} = nothing, verbose::Bool = false, solver::String = "lsqr", tol:.Float64b= 1e-8 )::Nothing

This function computes the approximation for the regularization parameter ``\lambda``.
"""
# parameter tol used only for lsqr
function approximate(
    a::approx,
    λ::Float64;
    max_iter::Int = 50,
    weights::Union{Vector{Float64},Nothing} = nothing,
    verbose::Bool = false,
    solver::String = "lsqr",
    tol::Float64 = 1e-8,
    nodeweights::Union{Vector{Float64},Nothing} = nothing,
)::Nothing
    M = size(a.X, 2)
    nf = get_NumFreq(a.trafo.setting)

    w = ones(Float64, nf)
    nw = ones(Float64, length(a.y))

    if !isnothing(weights)
        if (length(weights) != nf) || (minimum(weights) < 1)
            error("Weight requirements not fulfilled.")
        else
            w = weights
        end
    end

    if !isnothing(nodeweights)
        if (length(nodeweights) != length(a.y))
            error("The length of the nodeweights Vector doesnt match the Data.")
        else
            nw = nodeweights
        end
    end

    if (a.basis == "per" || a.basis == "expcos")
        what = GroupedCoefficients(a.trafo.setting, complex(w))
    else
        what = GroupedCoefficients(a.trafo.setting, w)
    end

    λs = collect(keys(a.fc))
    tmp = zeros(types[a.basis], nf)

    if length(λs) != 0
        idx = argmin(λs .- λ)
        tmp = copy(a.fc[λs[idx]].data)
    end

    if solver == "lsqr"
        diag_w_sqrt = sqrt(λ) .* sqrt.(w)
        if a.basis == "per"
            F_vec = LinearMap{ComplexF64}(
                fhat -> vcat(
                    sqrt.(nw).*(a.trafo * GroupedCoefficients(a.trafo.setting, fhat)),
                    diag_w_sqrt .* fhat,
                ),
                f -> vec(a.trafo' * (sqrt.(nw).*f[1:M])) + diag_w_sqrt .* f[M+1:end],
                M + nf,
                nf,
            )
            lsqr!(
                tmp,
                F_vec,
                vcat(sqrt.(nw).*a.y, zeros(ComplexF64, nf)),
                maxiter = max_iter,
                verbose = verbose,
                atol = tol,
                btol = tol,
            )
            a.fc[λ] = GroupedCoefficients(a.trafo.setting, tmp)
        else
            F_vec = LinearMap{Float64}(
                fhat -> vcat(
                    a.trafo * GroupedCoefficients(a.trafo.setting, fhat),
                    diag_w_sqrt .* fhat,
                ),
                f -> vec(a.trafo' * f[1:M]) + diag_w_sqrt .* f[M+1:end],
                M + nf,
                nf,
            )
            lsqr!(
                tmp,
                F_vec,
                vcat(a.y, zeros(Float64, nf)),
                maxiter = max_iter,
                verbose = verbose,
                atol = tol,
                btol = tol,
            )
            a.fc[λ] = GroupedCoefficients(a.trafo.setting, tmp)
        end
    elseif solver == "fista"
        ghat = GroupedCoefficients(a.trafo.setting, tmp)
        fista!(ghat, a.trafo, a.y, λ, what, max_iter = max_iter)
        a.fc[λ] = ghat
    else
        error("Solver not found.")
    end

    return
end

@doc raw"""
    approximate( a::approx; lambda::Vector{Float64} = exp.(range(0, 5, length = 5)), max_iter::Int = 50, weights::Union{Vector{Float64},Nothing} = nothing, verbose::Bool = false, solver::String = "lsqr" )::Nothing

This function computes the approximation for the regularization parameters contained in `lambda`.
"""
function approximate(
    a::approx;
    lambda::Vector{Float64} = exp.(range(0, 5, length = 5)),
    args...,
)::Nothing
    sort!(lambda, lt = !isless) # biggest λ will be computed first such that the initial guess 0 is somewhat good
    for λ in lambda
        approximate(a, λ; args...)
    end
    return
end

@doc raw"""
    evaluate( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Vector{ComplexF64},Vector{Float64}}

This function evaluates the approximation on the nodes `X` for the regularization parameter `λ`.
"""
function evaluate(
    a::approx,
    X::Matrix{Float64},
    λ::Float64,
)::Union{Vector{ComplexF64},Vector{Float64}}
    basis = a.basis

    if (basis == "per") && ((minimum(X) < -0.5) || (maximum(X) >= 0.5))
        error("Nodes need to be between -0.5 and 0.5.")
    elseif (basis == "cos") && ((minimum(X) < 0) || (maximum(X) > 1))
        error("Nodes need to be between 0 and 1.")
    elseif (basis == "cheb") && ((minimum(X) < -1) || (maximum(X) > 1))
        error("Nodes need to be between -1 and 1.")
    end

    Xt = copy(X)

    if basis == "cos"
        Xt ./= 2
    elseif basis == "cheb"
        Xt = acos.(Xt)
        Xt ./= 2 * pi
    elseif basis == "std"
        Xt ./= sqrt(2)
        Xt = erf.(Xt)
        Xt .+= 1
        Xt ./= 4
    end

    trafo = GroupedTransform(gt_systems[basis], a.U, a.N, Xt, a.dcos)
    return trafo * a.fc[λ]
end

@doc raw"""
    evaluate( a::approx; λ::Float64 )::Union{Vector{ComplexF64},Vector{Float64}}

This function evaluates the approximation on the nodes `a.X` for the regularization parameter `λ`.
"""
function evaluate(a::approx, λ::Float64)::Union{Vector{ComplexF64},Vector{Float64}}
    return a.trafo * a.fc[λ]
end

@doc raw"""
    evaluate( a::approx; X::Matrix{Float64} )::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}

This function evaluates the approximation on the nodes `X` for all regularization parameters.
"""
function evaluate(
    a::approx,
    X::Matrix{Float64},
)::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}
    return Dict(λ => evaluate(a, X, λ) for λ in collect(keys(a.fc)))
end

@doc raw"""
    evaluate( a::approx )::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}

This function evaluates the approximation on the nodes `a.X` for all regularization parameters.
"""
function evaluate(a::approx)::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}
    return Dict(λ => evaluate(a, λ) for λ in collect(keys(a.fc)))
end

@doc raw"""
    evaluateANOVAterms( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Matrix{ComplexF64},Matrix{Float64}}

This function evaluates the single ANOVA terms of the approximation on the nodes `X` for the regularization parameter `λ`.
"""
function evaluateANOVAterms(
    a::approx,
    X::Matrix{Float64},
    λ::Float64,
)::Union{Matrix{ComplexF64},Matrix{Float64}}

    basis = a.basis

    if (basis == "per") && ((minimum(X) < -0.5) || (maximum(X) >= 0.5))
        error("Nodes need to be between -0.5 and 0.5.")
    elseif (basis == "cos") && ((minimum(X) < 0) || (maximum(X) > 1))
        error("Nodes need to be between 0 and 1.")
    elseif (basis == "cheb") && ((minimum(X) < -1) || (maximum(X) > 1))
        error("Nodes need to be between -1 and 1.")
    end

    Xt = copy(X)

    if basis == "cos"
        Xt ./= 2
    elseif basis == "cheb"
        Xt = acos.(Xt)
        Xt ./= 2 * pi
    elseif basis == "std"
        Xt ./= sqrt(2)
        Xt = erf.(Xt)
        Xt .+= 1
        Xt ./= 4
    end
    
    if (basis == "per") # return matrix of size N (number data points) times number of ANOVA terms
        values = zeros(ComplexF64, size(Xt)[2], length(a.U))
    else
        values = zeros(Float64, size(Xt)[2], length(a.U))
    end
    
    trafo = GroupedTransform(gt_systems[basis], a.U, a.N, Xt, a.dcos)

    for j=1:length(a.U)
        u = a.U[j]
        values[:,j] = trafo[u] * a.fc[λ][u]
    end

    return values
end

@doc raw"""
    evaluateANOVAterms( a::approx; X::Matrix{Float64} )::Dict{Float64,Union{Matrix{ComplexF64},Matrix{Float64}}}

This function evaluates the single ANOVA terms of the approximation on the nodes `X` for all regularization parameters.
"""
function evaluateANOVAterms(
    a::approx,
    X::Matrix{Float64},
)::Dict{Float64,Union{Matrix{ComplexF64},Matrix{Float64}}}
    return Dict(λ => evaluateANOVAterms(a, X, λ) for λ in collect(keys(a.fc)))
end

@doc raw"""
    evaluateSHAPterms( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Matrix{ComplexF64},Matrix{Float64}}

This function evaluates for each dimension the Shapley contribution to the overall approximation on the nodes `X` for the regularization parameter `λ`.
"""
function evaluateSHAPterms(
    a::approx,
    X::Matrix{Float64},
    λ::Float64,
)::Union{Matrix{ComplexF64},Matrix{Float64}}

    basis = a.basis

    if (basis == "per") && ((minimum(X) < -0.5) || (maximum(X) >= 0.5))
        error("Nodes need to be between -0.5 and 0.5.")
    elseif (basis == "cos") && ((minimum(X) < 0) || (maximum(X) > 1))
        error("Nodes need to be between 0 and 1.")
    elseif (basis == "cheb") && ((minimum(X) < -1) || (maximum(X) > 1))
        error("Nodes need to be between -1 and 1.")
    end
    
    d = size(X)[1]
    
    if (basis == "per") # return matrix of size N (number of data points) times d (dimension)
        values = zeros(ComplexF64, size(X)[2], d)
    else
        values = zeros(Float64, size(X)[2], d)
    end
    
    terms = evaluateANOVAterms(a, X, λ) # evaluates all ANOVA terms at the nodes X

    for i=1:d
        for j=1:length(a.U)
            u = a.U[j]
            if (i in u)
                values[:,i] += terms[:,j]./length(u) # ANOVA terms are just equally distributed among the involved dimensions
            end
        end
    end

    return values
end

@doc raw"""
    evaluateSHAPterms( a::approx; X::Matrix{Float64} )::Dict{Float64,Union{Matrix{ComplexF64},Matrix{Float64}}}
    
This function evaluates for each dimension the Shapley contribution to the overall approximation on the nodes `X` for all regularization parameters.
"""
function evaluateSHAPterms(
    a::approx,
    X::Matrix{Float64},
)::Dict{Float64,Union{Matrix{ComplexF64},Matrix{Float64}}}
    return Dict(λ => evaluateSHAPterms(a, X, λ) for λ in collect(keys(a.fc)))
end
