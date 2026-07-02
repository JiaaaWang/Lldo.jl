"""
    define boxset C, which can model A(x) ∈ C
"""

module BoxSetUtils

using LinearAlgebra

export BoxSet, proj_C, dist_C, dist_C_inf, indicator_C

struct BoxSet
    name::Symbol
    lower::Vector{Float64} # can be -Inf for no lower bound
    upper::Vector{Float64} # can be +Inf for no upper bound  
end

function proj_C(
    z::AbstractVector{T}, 
    C::BoxSet
) where T<:Real

    lower_bound = C.lower
    upper_bound = C.upper

    @assert length(z) == length(lower_bound) "Point dimension must match box dimension"
    
    result = similar(z)

    for i in eachindex(z)
        if isinf(lower_bound[i])
            # only upper bound: (-∞, upper]
            result[i] = min(z[i], upper_bound[i])
        elseif isinf(upper_bound[i])
            # only lower bound: [lower, ∞)
            result[i] = max(z[i], lower_bound[i])
        else
            # both bounds: [lower, upper]
            result[i] = min(max(z[i], lower_bound[i]), upper_bound[i])
        end
    end

    return result
end

function dist_C(
    z::AbstractVector{T}, 
    C::BoxSet
) where T<:Real

    proj_z = proj_C(z, C::BoxSet)
    return norm(z - proj_z)
end

function dist_C_inf(
    z::AbstractVector{T}, 
    C::BoxSet
) where T<:Real

    proj_z = proj_C(z, C::BoxSet)
    return norm(z - proj_z, Inf)
end

function indicator_C(
    z::AbstractVector{T}, 
    C::BoxSet
) where T<:Real

    dist_z = dist_C(z, C)
    return dist_z < 1e-8 ? 0.0 : Inf
end

end # module