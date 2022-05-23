function regkernel(
    v,
    op,
    U,
    weights,
    lambda,
)
    z = op' * v
    j = 1
    for u in U
        if u == []
            z[u] = weights[1:length(z[u])].^(-1) .* z[u]
            j = j + 1
        else
            z[u] = weights[j:j+length(z[u])-1].^(-1) .* z[u]
            j = j + length(z[u])
        end
    end
    z = op * z
    z = z + lambda * v
    return z #vec(z)
end
