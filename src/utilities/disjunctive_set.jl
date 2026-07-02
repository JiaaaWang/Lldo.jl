"""
    the projection to the set D, where F_j(x) ∈ D
"""

module DisjunctiveSetUtils

using LinearAlgebra

export proj_D, dist_D, indicator_D, proj_CC, proj_CC_Tb

function proj_D(z::AbstractVector{T}) where T<:Real
    z_proj = copy(z)
    
    has_nonpositive = any(z .<= 0) # check if there are any non-positive elements, true or false
    
    if has_nonpositive 
        # case 1:
        for i in eachindex(z_proj)
            if z_proj[i] <= 0
                z_proj[i] = 0.0
            end
        end
    else

        # case 2: if all elements are positive, set the smallest one to 0, can have multi candidates, but only set the first one to 0
        min_index = argmin(z)
        z_proj[min_index] = 0.0
    end
    
    return z_proj
end

function dist_D(
    z::AbstractVector{T}
) where T<:Real
    z_proj = proj_D(z)
    return norm(z - z_proj)
end

function indicator_D(
    z::AbstractVector{T}
) where T<:Real

    dist_z = dist_D(z)
    return dist_z < 1e-8 ? 0.0 : Inf
end

function proj_CC(f::AbstractVector{T}, g::AbstractVector{T}) where T<:Real

    f_proj = similar(f)
    g_proj = similar(g)
    
    for i in eachindex(f, g)

        if f[i] <= 0 || g[i] <= 0

            f_proj[i] = max(f[i], 0)
            g_proj[i] = max(g[i], 0)

        else

            if f[i] < g[i]
                f_proj[i] = 0
                g_proj[i] = g[i]
            elseif g[i] < f[i]
                f_proj[i] = f[i]
                g_proj[i] = 0
            else
                f_proj[i] = 0
                g_proj[i] = g[i]
            end
            
        end

    end

    return f_proj, g_proj
end

#  special projection for smoothed region Tβ 
function proj_CC_Tb(
    f::AbstractVector{T}, 
    g::AbstractVector{T},
    β::Vector{T}
) where T<:Real

    f_proj = similar(f)
    g_proj = similar(g)
    
    for i in eachindex(f, g)

        if f[i] <= 0 || g[i] <= 0

            f_proj[i] = max(f[i], 0)
            g_proj[i] = max(g[i], 0)
        
        elseif g[i] >= f[i] / (1 - β[i])

            f_proj[i] = 0
            g_proj[i] = g[i]

        elseif g[i] <= (1 - β[i]) * f[i]

            f_proj[i] = f[i]
            g_proj[i] = 0

        else # Tβ: g[i] > (1 - β[i])* f[i] && g[i] < f[i] / (1 - β[i])

            f_proj[i] = 0
            g_proj[i] = 0
            
        end

    end

    return f_proj, g_proj
end

end # module