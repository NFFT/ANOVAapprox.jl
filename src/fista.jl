function bisection(
    fun,
    fval,
    left,
    right,
    fleft,
    fright;
    max_iter = 10,
    tol = 1e-15,
    verbose = false,
)
    fright -= fval
    fleft -= fval

    for iter = 1:max_iter
        global middle = (left + right) / 2
        fmiddle = fun(middle) .- fval

        if sign(fmiddle) == sign(fleft)
            left = middle
            fleft = fmiddle
        else
            right = middle
            fright = fmiddle
        end
        verbose && println("residual for Bisection $(abs(fmiddle))")
        abs(fmiddle) < tol && break
    end
    return middle
end

function newton(fun, dfun, fval, x; max_iter = 10, tol = 1e-15, verbose = false)
    for iter = 1:max_iter
        f = fun(x)
        df = dfun(x)
        (isnan(f) || isnan(df) || abs(df) < 1e-15) && break
        x += (fval - f) / df
        verbose && println(
            "residual for Newton: $(abs(f-fval))\nf: $(f) df: $(df) fval: $(fval) x $(x)",
        )
        abs(f - fval) < tol && break
    end
    return x
end

function λ2ξ(λ, what, y; verbose = false)
    fun = ξ -> sum(abs.(what .* (y ./ (1 / ξ .+ what)) .^ 2))
    dfun = ξ -> 2 * sum(abs.(what .* (y ./ (1 / ξ .+ what)) .^ 2 ./ (1 / ξ .+ what))) * ξ^-2

    fright = sum(abs.(what .* (y ./ (1 .+ what)) .^ 2))
    if λ^2 < fright
        fleft = 0
        ξ = bisection(
            fun,
            λ^2,
            1e-10,
            1,
            fleft,
            fright;
            max_iter = 25,
            tol = 1e-10,
            verbose = verbose,
        )
    else
        fleft = sum(what .* (y ./ what) .^ 2)
        ξ =
            1 / bisection(
                ξ -> fun(1 / ξ),
                λ^2,
                1e-10,
                1,
                fleft,
                fright;
                max_iter = 25,
                tol = 1e-16,
                verbose = verbose,
            )
        ξ > 100 && return ξ
    end

    # apply Newton on f(exp(x)). Small solutions can be found more accurate this way and we work our way around negative solutions.
    ξ =
        newton(
            x -> fun(exp(x)),
            x -> dfun(exp(x)) * exp(x),
            λ^2,
            log(ξ);
            max_iter = 50,
            tol = 1e-16,
            verbose = verbose,
        ) |> exp

    if abs(fun(ξ) - λ^2) > 1
        error("λ2ξ: something went wrong minimizing. (residual: $(abs(fun(ξ)-λ^2))")
    end

    return ξ
end

function loss2_function(x)
    if x > 1
        return 0
    else
        return (1 - x)^2
    end
end

function ∇loss2_function(x)
    # derivative of quadratic loss:
    if x > 1
        return 0
    else
        return (1 - x) * -2
    end
end

function fista!(
    ghat::GroupedCoefficients,
    F::GroupedTransform,
    y::Union{Vector{ComplexF64},Vector{Float64}},
    λ::Float64,
    what::GroupedCoefficients;
    L = "adaptive",
    max_iter::Int = 25,
    classification::Bool = false,
)
    adaptive = (L == "adaptive")
    if adaptive
        L = 1
        η::Int = 2
    end

    U = [s[:u] for s in ghat.setting]

    hhat = GroupedCoefficients(ghat.setting, copy(vec(ghat)))
    t = 1.0
    if classification
        val = [
            (1 / length(y)) * sum(loss2_function.(y .* (F * hhat))) +
            λ * sum(abs.(hhat.data)),
        ]
    else
        val = [norm((F * hhat) - y)^2 / 2 + λ * sum(norms(hhat, what))]
    end

    for k = 1:max_iter-1
        ghat_old = GroupedCoefficients(ghat.setting, copy(vec(ghat)))
        t_old = t

        if classification
            fgrad = F' * (1 / length(y) * y .* (∇loss2_function.(y .* (F * hhat)))) #TODO: ghat or hhat
        else
            Fhhat = F * hhat
            fgrad = (F' * (Fhhat - y))
        end
        while true
            # p_L(hhat)
            if classification
                for k = 1:length(ghat.data)
                    #fhat > 0:
                    if L * hhat[k] - fgrad[k] > λ
                        ghat[k] = hhat[k] - 1 / L * fgrad[k] - 1 / L * λ
                        #fhat < 0:
                    elseif fgrad[k] - L * hhat[k] > λ
                        ghat[k] = hhat[k] - 1 / L * fgrad[k] + 1 / L * λ
                    else
                        ghat[k] = 0.0
                    end
                end
            else
                set_data!(ghat, vec(hhat - 1 / L * fgrad))

                mask = map(u -> (λ / L)^2 < sum(abs.(ghat[u] .^ 2 ./ what[u])), U)
                ξs = map(fetch, map(u -> @spawn(λ2ξ(λ / L, what[u], ghat[u])), U[mask]))
                #ξs = pmap(u -> λ2ξ(λ / L, what[u], ghat[u]), U[mask])
                for u in U[.!mask]
                    ghat[u] = 0 * ghat[u]
                end
                for (u, ξ) in zip(U[mask], ξs)
                    if ξ == Inf
                        ghat[u] = 0 * ghat[u]
                    else
                        ghat[u] = ghat[u] ./ (1 .+ ξ * what[u])
                    end
                end
            end


            if !adaptive
                if classification
                    append!(
                        val,
                        (1 / length(y)) * sum(loss2_function.(y .* (F * hhat))) +
                        λ * sum(abs.(hhat.data)),
                    )
                else
                    append!(val, norm((Fhhat) - y)^2 / 2 + λ * sum(norms(hhat, what)))
                end
                break
            end

            # F
            if classification
                Fvalue =
                    (1 / length(y)) * sum(loss2_function.(y .* (F * ghat))) +
                    λ * sum(abs.(ghat.data))
            else
                Fvalue = norm((F * ghat) - y)^2 / 2 + λ * sum(norms(ghat, what))
            end

            # Q
            if classification
                Q = (
                    (1 / length(y)) * sum(loss2_function.(y .* (F * hhat))) +
                    dot(vec(ghat - hhat), vec(fgrad)) +
                    L / 2 * norm(vec(ghat - hhat))^2 +
                    λ * sum(abs.(vec(ghat)))
                )
            else
                Q = (
                    norm((Fhhat) - y)^2 / 2 +
                    dot(vec(ghat - hhat), vec(fgrad)) +
                    L / 2 * norm(vec(ghat - hhat))^2 +
                    λ * sum(norms(ghat, what))
                )
            end


            if real(Fvalue) < real(Q) + 1e-10 || L >= 2^32
                append!(val, Fvalue)
                break
            else
                L *= η
            end
        end

        # update t
        t = (1 + sqrt(1 + 4 * t^2)) / 2

        # update hhat
        hhat = ghat + (t_old - 1) / t * (ghat - ghat_old)

        # stoping criteria
        resnorm = norm(vec(ghat_old - ghat))
        resnorm < 1e-16 && break
        abs(val[end] - val[end-1]) < 1e-16 && break
    end
end
