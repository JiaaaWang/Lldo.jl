"""
    Lasry-Lions double envelope
"""

module VCCsUtils

export LL_env_grad, LL_prox, find_M_practice

# identify the index set M (geometry perspective)
function find_M_practice(z::Vector{Float64}, λ::Float64, μ::Float64)
    p = length(z)
    sorted_vals, sorted_idx = sort(z), sortperm(z) # sort coordinates, but keep original indices
    S = cumsum(sorted_vals) # precompute prefix sums
    G(m) = λ * S[m] - (m*λ - μ) * sorted_vals[m] # define G(m)

    # Case 1: m = p
    if G(p) >= 0
        return sorted_idx[1:p]   # all indices included
    end

    # Case 2: find crossing m with G(m) ≥ 0 and G(m+1) < 0
    for m in 1:p-1
        if G(m) >= 0 && G(m+1) < 0
            return sorted_idx[1:m]   # first m indices are M
        end
    end

    # If no M found, return nothing (should not happen mathematically)
    return nothing
end

# identify the index set M (disjunctive perspective)

# function find_M_practice(z::Vector{Float64}, λ::Float64, μ::Float64)
#     p = length(z)
#     sorted_vals, sorted_idx = sort(z), sortperm(z) 
#     S = cumsum(sorted_vals) 
#     G(m) = (λ * S[m]) / (m*λ - μ) 

        
#     if G(p) - sorted_idx[p] >= 0
#         return sorted_idx[1:p]   
#     end

#     for m in 1 : p-1
#         if G(m) - sorted_idx[m] >= 0 && G(m) - sorted_idx[m+1] < 0
#             return sorted_idx[1:m]   
#         end
#     end

#     return nothing
# end

# calculate the Lasry-Lions double envelope and its gradient
function LL_env_grad(z::Vector{Float64}, λ::Float64, μ::Float64)
    p = length(z)

    # basic validation
    @assert p >= 2 "p must be >= 2"
    @assert λ > μ > 0 "require λ > μ > 0"

    # initialize gradient
    grad = zeros(Float64, p)

    # if there are nonpositive entries 
    neg_flag = z .<= 0.0
    if any(neg_flag)
        neg_index = findall(neg_flag) # indices with nonpositive z
        s = sum(z[i]^2 for i in neg_index)
        env = s / (2 * (λ - μ))

        # gradient: [∇env]_i = (1/(λ-μ)) * z_i for i in I_<0(z), else 0
        for i in neg_index
            grad[i] = z[i] / (λ - μ)
        end
        return env, grad
    end

    # otherwise all z >= 0 
    M = find_M_practice(z, λ, μ)
    m = length(M)

    # compute sums over M
    sum_M = sum(z[i] for i in M)
    sumsq_M = sum(z[i]^2 for i in M)

    denominator = m * λ - μ
    env = (λ * (sum_M^2)) / (2 * μ * denominator) - sumsq_M / (2 * μ)

    # gradient: for i in M: grad[i] = λ/( μ (m(λ-μ)) ) * sum_M - (1/μ) * z[i]
    factor = λ / (μ * denominator)
    for i in M
        grad[i] = factor * sum_M - z[i] / μ
    end

    return env, grad
end

# calculate the Lasry-Lions double envelope's prox
function LL_prox(z::Vector{Float64}, λ::Float64, μ::Float64)
    p = length(z)

    # basic validation
    @assert p >= 2 "p must be >= 2"
    @assert λ > μ > 0 "require λ > μ > 0"

    # initialize prox
    prox = z

    # if there are nonpositive entries 
    neg_flag = z .<= 0.0
    if any(neg_flag)
        neg_index = findall(neg_flag) # indices with nonpositive z
        for i in neg_index
            prox[i] = (λ * z[i]) / (λ - μ)
        end
        return prox
    end

    # otherwise all z > 0 
    M = find_M_practice(z, λ, μ)
    m = length(M)

    # compute sums over M
    sum_M = sum(z[i] for i in M)
    denominator = m * λ - μ
    factor = λ / denominator
    for i in M
        prox[i] = factor * sum_M
    end

    return prox
end

end # module VCCs

# a tailored module for CCs
module CCsUtils

    export LL_double_env_compl, grad_LL_double_env_compl

    function LL_double_env_compl(f, g, lambda, mu)
        
        index1 = (f .<= 0) .& (g .<= 0)
        index2 = (f .> 0) .& (g .<= ((lambda .- mu) ./ lambda) .* f)
        index3 = (g .> 0) .& (f .<= ((lambda .- mu) ./ lambda) .* g)
        index4 = (f .> 0) .& (g .< (lambda ./ (lambda .- mu)) .* f) .& 
                 (g .> ((lambda .- mu) ./ lambda) .* f)
        
        e1 = ((f.^2 .+ g.^2) ./ (2 .* (lambda .- mu))) .* index1
        e2 = ((g.^2) ./ (2 .* (lambda .- mu))) .* index2
        e3 = ((f.^2) ./ (2 .* (lambda .- mu))) .* index3
        e4 = (lambda .* (f .+ g).^2 ./ (2 .* mu .* (2 .* lambda .- mu)) .- 
            (f.^2 .+ g.^2) ./ (2 .* mu)) .* index4
        
        return e1 .+ e2 .+ e3 .+ e4
    end

    function grad_LL_double_env_compl(f, g, lambda, mu)
        
        index1 = (f .<= 0) .& (g .<= 0)
        index2 = (f .> 0) .& (g .<= ((lambda .- mu) ./ lambda) .* f)
        index3 = (g .> 0) .& (f .<= ((lambda .- mu) ./ lambda) .* g)
        index4 = (f .> 0) .& (g .< (lambda ./ (lambda .- mu)) .* f) .& 
                 (g .> ((lambda .- mu) ./ lambda) .* f)
        
        g11 = (f ./ (lambda .- mu)) .* index1
        g12 = (g ./ (lambda .- mu)) .* index1

        g21 = zeros(size(f)) .* index2
        g22 = (g ./ (lambda .- mu)) .* index2

        g31 = (f ./ (lambda .- mu)) .* index3
        g32 = zeros(size(g)) .* index3
        
        g41 = (lambda .* (f .+ g) ./ (mu .* (2 .* lambda .- mu)) .- f ./ mu) .* index4
        g42 = (lambda .* (f .+ g) ./ (mu .* (2 .* lambda .- mu)) .- g ./ mu) .* index4
        
        g1 = g11 .+ g21 .+ g31 .+ g41
        g2 = g12 .+ g22 .+ g32 .+ g42
        
        return g1, g2
    end

end # module CCs