
using Distributions
using StatsBase
##### fuctions that trasforms data X dimension-wise
#### transform data to [-1/2,1/2]^d ####
### input:
# X     .... Matrix d x M
# sigma .... if sigma should be output
# X_test .... if test data should be transformed: Insert X_test in format d x M_test
function transform_cube(X; sigma =false, X_test =false)
    d = size(X,1)
    M = size(X,2)
    if X_test ==false
        X_trafo = zeros(size(X))
    else
        X_trafo = zeros(size(X_test))
    end
    σ = Dict()
    for i =1:d
        σ[i] = 1.06 * minimum([std(vec(X[i,:])),iqr(vec(X[i,:]))/1.34])*M^(-1/5)  ###todo: sigma auch als output
        if X_test == false
            X_trafo[i,:] = reshape(Rho_circ(X[i,:],X[i,:]',mm,σ[i]) .-0.5,1,:)
        else
            X_trafo[i,:] = reshape(Rho_circ(X_test[i,:],X[i,:]',mm,σ[i]) .-0.5,1,:)
        end
    end
    if sigma == true
        return X_trafo,σ
    else
        return X_trafo
    end
end


#### transform data to [-1/2,1/2]^d ####
# input: ####
# X         .... Matrix d x M
# optional: ####
# sigma     .... if sigma should be output
# X_test    .... if test data should be transformed: Insert X_test in format d x M_test
# KDE       .... choice of parameter selction: ROT or DPI
function transform_R(X; sigma = false,  X_test =false, KDE = "ROT")
    d = size(X,1)
    M = size(X,2)
    if X_test == false
        X_trafo = zeros(size(X))
    else
        X_trafo = zeros(size(X_test))
    end
    σ = Dict()
    for i =1:d
        if KDE == "ROT"
            σ[i] = 1.06 * minimum([std(vec(X[i,:])),iqr(vec(X[i,:]))/1.34])*M^(-1/5)
        elseif KDE == "DPI"
            Ψ_8 = 105/(32*sqrt(π)*std(vec(X[i,:]))^9)
            g_1 = (-(-30/sqrt(2*π))/(Ψ_8*M))^(1/9)
            Ψ_6 = 1/(M^2*g_1^7) * sum(rho_diff6((X[i,:] .-X[i,:]')./g_1))
            g_2 = ((-6/sqrt(2*π))/(Ψ_6*M))^(1/7)
            Ψ_4 = 1/(M^2*g_2^5) * sum(rho_diff4((X[i,:] .-X[i,:]')./g_2))
            σ[i] = (1/(2*sqrt(π)*Ψ_4*M))^(1/5)
        else
            error("KDE not defined. Choose from ROT or DPI")
        end
        ρ = MixtureModel(Vector{Normal}(vec( Normal.(X[i,:], σ[i]))))
        if X_test == false
            X_trafo[i,:] = cdf.(ρ, X[i,:]).-0.5                     # omit -0.5 for data on [0,1]
        else
            X_trafo[i,:] = cdf.(ρ, X_test[i,:]).-0.5                # omit -0.5 for data on [0,1]
            #### todo: alles außerhalb von [-0.5,0.5 ] auf +-0.5 setzen
        end
    end
    if sigma == true
        return X_trafo,σ
    else
        return X_trafo
    end
end





#### help functions for KDE on cube:
mm = 3                          #order of kernel B-Spline
function Rho1(x,m)              #integral of one B-Spline - kernel
    if m == 3
        if x<-1.5
            return 0
        elseif x<-0.5
            return 9/8*x + 3/4*x^2 +1/6*x^3 + 9/16
        elseif x<0.5
            return 1/6+ 3/4*x- 1/3*x^3+1/3
        elseif x<1.5
            return 5/6 + 9/8*x - 3/4*x^2 +1/6*x^3 -19/48
        else
            return 1
        end
    end
end
function Rho_circ(x,X,m,σ)
    if m == 3
        M = size(X,2)
        return 1/(M) * sum(Rho1.((x .- X)/σ,m) ,dims=2)
    end
end


# help functions for KDE on R:
function rho_diff6(X)
    return pdf.(Normal(0,1),X) .*(X.^6 - 15*X.^4 +45* X.^2 .-15)
end
function rho_diff4(X)
    return pdf.(Normal(0,1),X) .*(X.^4 - 6*X.^2 .+3)
end
