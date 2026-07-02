""" 

Algencan and Scholtes:

    For VCCs:
        constraints: F := { F_1,…, F_r }
        
        subproblem:
        min  f(x)
        s.t. A(x) ∈ C,
             F_1(x) ≥ 0, 
             F_{1 1} * ⋯ * F_{1 p_{1}} ≤ τ,
             ⋮ 
             F_r(x) ≥ 0,
             F_{r 1} * ⋯ * F_{r p_{r}} ≤ τ

    For CCs:
        constraints: F := { G, H }
        
        subproblem:
        min  f(x)
        s.t. A(x) ∈ C,
             G(x) ≥ 0, 
             H(x) ≥ 0,
             <G(x), H(x)> ≤ τ

    where τ = 0 for Algencan, and τ > 0 for Scholtes

Ipopt for VCCs/CCs is underdeveloped

"""

module NLPSolvers

using LinearAlgebra
using Printf

using ..BoxSetUtils
using ..FunctionStruct
using ..AlgencanHelper
using ..IpoptHelper

export direct_algencan, scholtes, ScholtesOptions

function merge_constraints_CC(
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    τ::Real,
    m::Int, # dimension of A(x)
    r::Int # number of CCs
)
    
    G_CC = F[1]
    H_CC = F[2]

    function A_merge_func(x)

        value_G, _ = G_CC(x)
        value_H, _ = H_CC(x)
        value_dot = dot(value_G, value_H) - τ

        if m == 0
            return  vcat(value_G, value_H, value_dot)
        else
            value_A, _ = A(x)
            return  vcat(value_A, value_G, value_H, value_dot)
        end
        
    end

    function A_merge_jaco(x)

        value_G, jaco_G = G_CC(x)
        value_H, jaco_H = H_CC(x)
        jaco_dot = value_H' * jaco_G + value_G' * jaco_H

        if m == 0
            return vcat(jaco_G, jaco_H, jaco_dot)
        else
            _, jaco_A = A(x)
            return vcat(jaco_A, jaco_G, jaco_H, jaco_dot)
        end

    end

    A_merge = VectorFunction(:A_merge, A_merge_func, A_merge_jaco)
    
    if m == 0
        C_merge_lower = vcat(fill(0.0, 2 * r), -Inf)
        C_merge_upper = vcat(fill(Inf, 2 * r), 0.0)
        C_merge = BoxSet(:C_merge, C_merge_lower, C_merge_upper)
    else
        C_lower = C.lower
        C_upper = C.upper
        C_merge_lower = vcat(C_lower, fill(0.0, 2 * r), -Inf)
        C_merge_upper = vcat(C_upper, fill(Inf, 2 * r), 0.0)
        C_merge = BoxSet(:C_merge, C_merge_lower, C_merge_upper)
    end

    return A_merge, C_merge
end

function merge_constraints_VCC(
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    τ::Real,
    m::Int, # dimension of A(x)
    r::Int, # number of VCCs 
    p::Vector{<:Int} # dimension of each VCC
)

    function A_merge_func(x)

        value_F = Float64[]
        for i in eachindex(F)
            F_i = F[i]
            value_F_i, _ = F_i(x)
            value_dot_F_i = prod(value_F_i) - τ
            value_F = vcat(value_F, value_F_i, value_dot_F_i)
        end

        if m == 0
            return  value_F
        else
            value_A, _ = A(x)
            return  vcat(value_A, value_F)
        end
        
    end

    function A_merge_jaco(x)

        jaco_F = Matrix{Float64}(undef, 0, length(x))
        for i in eachindex(F)
            F_i = F[i]
            value_F_i, jaco_F_i = F_i(x)
            
            coeffs = similar(value_F_i)
            for k in eachindex(value_F_i)
                coeffs[k] = prod(value_F_i[j] for j in eachindex(value_F_i) if j != k)
            end

            jaco_dot_F_i = coeffs' * jaco_F_i   
            jaco_F = vcat(jaco_F, jaco_F_i, jaco_dot_F_i)
        end

        if m == 0
            return jaco_F
        else
            _, jaco_A = A(x)
            return vcat(jaco_A, jaco_F)
        end

    end

    A_merge = VectorFunction(:A_merge, A_merge_func, A_merge_jaco)
    
    C_merge_lower = Float64[]
    C_merge_upper = Float64[]

    if m == 0
        for i in 1 : r
            C_merge_lower = vcat(C_merge_lower, fill(0.0, p[i]), -Inf)
            C_merge_upper = vcat(C_merge_upper, fill(Inf, p[i]), 0.0)
        end
        
        C_merge = BoxSet(:C_merge, C_merge_lower, C_merge_upper)
    else
        C_lower = C.lower
        C_upper = C.upper
        C_merge_lower = vcat(C_merge_lower, C_lower)
        C_merge_upper = vcat(C_merge_upper, C_upper)

        for i in 1 : r
            C_merge_lower = vcat(C_merge_lower, fill(0.0, p[i]), -Inf)
            C_merge_upper = vcat(C_merge_upper, fill(Inf, p[i]), 0.0)
        end

        C_merge = BoxSet(:C_merge, C_merge_lower, C_merge_upper)
    end

    return A_merge, C_merge
end

# using algencan to solve mpvcc directly 
function direct_algencan(
    f::ScalarFunction,
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x0::AbstractVector,
    m::Int, # dimension of A(x)
    r::Int; # number of VCCs or CCs
    tol_dual::Real = 1e-6,
    tol_prim::Real = 1e-6,
    prob_type::String = "vcc"
)

    algencan_start_time = time()

    if prob_type == "cc"
        
        A_merge, C_merge = merge_constraints_CC(F, A, C, 0.0, m, r)

    elseif prob_type == "vcc"

        p = Vector{Int}()
        for i in eachindex(F)
            F_i = F[i] 
            value_F_i, _ = F_i(x0)
            push!(p, length(value_F_i))
        end
        
        A_merge, C_merge = merge_constraints_VCC(F, A, C, 0.0, m, r, p)

    else
        error("Unknown problem type: '$prob_type'. Use 'cc' or 'vcc'")
    end

    inner_result = algencan_solver(
        f,
        A_merge,
        C_merge,
        x0;
        tol_dual = tol_dual,
        tol_prim = tol_prim
    )

    algencan_elapsed_time = time() - algencan_start_time

    x_opt = inner_result.x
    f_opt = inner_result.f_opt
    algencan_V = inner_result.cons_vio
    cons_vio_inf = inner_result.cons_vio_inf
    algencan_solved = (inner_result.status == :first_order || inner_result.status == :acceptable) && (cons_vio_inf <= tol_prim)

    # check the stationarity of returned point 
    algencan_status = 
    if algencan_solved 
        :sta
    elseif !algencan_solved && (cons_vio_inf <= tol_prim)
        :fea
    elseif !algencan_solved && (cons_vio_inf > tol_prim)
        :infea
    else
        :unknown
    end

    println("\nAlgencan progress summary:")
println("==============================================================================================")
    @printf("%-12s %-15s %-18s %-15s %-15s %-15s\n", 
        "Time", "Objective", "Status", "Tolerance", "Max violation", "Infea measure")
    println("----------------------------------------------------------------------------------------------")
    @printf("%-12s %-15s %-18s %-15s %-15s %-15s\n",
        "$(round(algencan_elapsed_time, digits=2))s", 
        "$(@sprintf "%.2e" f_opt)", 
        "$algencan_status",
        "$(@sprintf "%.2e" tol_dual)",
        "$(@sprintf "%.2e" cons_vio_inf)",
        "$(@sprintf "%.2e" algencan_V)")
println("==============================================================================================")

    return (;  
        f_opt = f_opt,           
        cons_vio = algencan_V,
        cons_vio_inf = cons_vio_inf,
        time = algencan_elapsed_time,
        status = algencan_status,
        x = copy(x_opt)
    )
end

# using ipopt to solve mpvcc directly 
function direct_ipopt(
    f::ScalarFunction,
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x0::AbstractVector,
    m::Int, # dimension of A(x)
    r::Int; # number of VCCs or CCs
    tol_dual::Real = 1e-6,
    tol_prim::Real = 1e-6,
    prob_type::String = "vcc"
)

    ipopt_start_time = time()

    if prob_type == "cc"
        
        A_merge, C_merge = merge_constraints_CC(F, A, C, 0.0, m, r)

    elseif prob_type == "vcc"

        p = Vector{Int}()
        for i in eachindex(F)
            F_i = F[i] 
            value_F_i, _ = F_i(x0)
            push!(p, length(value_F_i))
        end
        
        A_merge, C_merge = merge_constraints_VCC(F, A, C, 0.0, m, r, p)

    else
        error("Unknown problem type: '$prob_type'. Use 'cc' or 'vcc'")
    end

    inner_result = ipopt_solver(
        f,
        A_merge,
        C_merge,
        x0;
        tol_dual = tol_dual,
        tol_prim = tol_prim
    )

    ipopt_elapsed_time = time() - ipopt_start_time

    x_opt = inner_result.x
    f_opt = inner_result.f_opt
    ipopt_V = inner_result.cons_vio
    cons_vio_inf = inner_result.cons_vio_inf

    ipopt_solved = (inner_result.status == :OPTIMAL || inner_result.status == :LOCALLY_SOLVED) && (cons_vio_inf <= tol_prim)

    # check the stationarity of returned point 
    ipopt_status = 
    if ipopt_solved 
        :sta
    elseif !ipopt_solved && (cons_vio_inf <= tol_prim)
        :fea
    elseif !ipopt_solved && (cons_vio_inf > tol_prim)
        :infea
    else
        :unknown
    end

    println("\nIpopt progress summary:")
println("==============================================================================================")
    @printf("%-12s %-15s %-18s %-15s %-15s %-15s\n", 
        "Time", "Objective", "Status", "Tolerance", "Max violation", "Infea measure")
    println("----------------------------------------------------------------------------------------------")
    @printf("%-12s %-15s %-18s %-15s %-15s %-15s\n",
        "$(round(ipopt_elapsed_time, digits=2))s", 
        "$(@sprintf "%.2e" f_opt)", 
        "$ipopt_status",
        "$(@sprintf "%.2e" tol_dual)",
        "$(@sprintf "%.2e" cons_vio_inf)",
        "$(@sprintf "%.2e" ipopt_V)")
println("==============================================================================================")

    return (;
        f_opt = f_opt,           
        cons_vio = ipopt_V,
        cons_vio_inf = cons_vio_inf,
        time = ipopt_elapsed_time,
        status = ipopt_status,
        x = copy(x_opt)
    )
end

# options struct
Base.@kwdef mutable struct ScholtesOptions{T<:Real}

    ε0::T = one(T) * 0.1
    τ0::T = one(T) * 1.0

    κ_ε::T = one(T) * 0.1         # decrease of subproblem tol. ε
    κ_τ::T = one(T) * 0.5         # decrease of relaxation parameter
    τ_min::T = one(T) * 1e-16     # lower bound of relaxation parameter

    tol_prim::T = one(T) * 1e-6
    tol_dual::T = one(T) * 1e-6

    prob_type::String = "vcc"

    verbose::Bool = true

end

# scholtes main function
function scholtes(
    f::ScalarFunction,
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x0::AbstractVector,
    m::Int, # dimension of A(x)
    r::Int; # number of VCCs or CCs
    options::ScholtesOptions = ScholtesOptions{Float64}()
)

    scholtes_start_time = time()

    # initialization
    opt = options

    x = copy(x0)
    τ = opt.τ0
    ε = opt.ε0

    scholtes_tot_it = 0
    scholtes_solved = false
    scholtes_sub_solved = false
    scholtes_tired = τ < opt.τ_min
    scholtes_V = nothing
    cons_vio_inf = nothing

    if opt.prob_type == "cc"

        A_merge_0, C_merge_0 = merge_constraints_CC(F, A, C, 0.0, m, r)

    elseif opt.prob_type == "vcc"

        p = Vector{Int}()
        for i in eachindex(F)
            F_i = F[i] 
            value_F_i, _ = F_i(x0)
            push!(p, length(value_F_i))
        end

        A_merge_0, C_merge_0 = merge_constraints_VCC(F, A, C, 0.0, m, r, p)
    else
        error("Unknown problem type: '$(opt.prob_type)'. Use 'cc' or 'vcc'")
    end

    # loop
    scholtes_can_stop = scholtes_solved || scholtes_tired

    if opt.verbose 
        println("Scholtes outer loop:")
        println("Iter |Inner flag  | Inner tolerance | Objective  | τ          | Infea measure  | ")
        println("-----|------------|-----------------|------------|------------|----------------|")
    end

    while !scholtes_can_stop

        scholtes_tot_it += 1

        if opt.prob_type == "cc"

            A_merge, C_merge = merge_constraints_CC(F, A, C, τ, m, r)

        elseif opt.prob_type == "vcc"

            A_merge, C_merge = merge_constraints_VCC(F, A, C, τ, m, r, p)

        else
            error("Unknown problem type: '$(opt.prob_type)'. Use 'cc' or 'vcc'")
        end

        inner_result = algencan_solver(
            f,
            A_merge,
            C_merge,
            x0;
            tol_dual = ε,
            tol_prim = opt.tol_prim
        )

        # record results
        scholtes_sub_solved = (inner_result.status == :first_order) || (inner_result.status == :acceptable)
        x = inner_result.x
        f_outer, _ = f(x)

        # calculate residual
        value_A_merge_0, _ = A_merge_0(x)
        scholtes_V = dist_C(value_A_merge_0, C_merge_0)
        cons_vio_inf = dist_C_inf(value_A_merge_0, C_merge_0)

        # print interim results
        if opt.verbose 
            @printf "%4d | %10s | %15.2e | %10.2e | %10.2e | %14.2e |\n" scholtes_tot_it scholtes_sub_solved ε f_outer τ scholtes_V
        end

        # termination checks
        scholtes_solved = (scholtes_sub_solved) && (cons_vio_inf <= opt.tol_prim)
        scholtes_tired = τ < opt.τ_min
        scholtes_can_stop = scholtes_solved || scholtes_tired

        # parameter update
        if !scholtes_can_stop
            
            # update relaxation parameter
            τ = τ * opt.κ_τ

            # update inner tolerance
            ε = max(opt.κ_ε * ε, opt.tol_dual)
        end

    end

    scholtes_elapsed_time = time() - scholtes_start_time
    f_opt, _ = f(x)
 
    # check the stationarity of returned point 
    scholtes_status = 
    if scholtes_solved 
        :sta
    elseif scholtes_tired && (cons_vio_inf <= opt.tol_prim)
        :min_tau_fea
    elseif scholtes_tired && (cons_vio_inf > opt.tol_prim)
        :min_tau_infea_sta
    else
        :unknown
    end

    println("\nScholtes progress summary:")
println("===========================================================================================================================")
    @printf("%-12s %-15s %-12s %-18s %-15s %-15s %-15s %-15s\n", 
        "Time", "Objective", "Outer iter", "Status", "Tolerance", "τ", "Max violation", "Infea measure")
    println("---------------------------------------------------------------------------------------------------------------------------")
    @printf("%-12s %-15s %-12d %-18s %-15s %-15s %-15s %-15s\n",
        "$(round(scholtes_elapsed_time, digits=2))s", 
        "$(@sprintf "%.2e" f_opt)", 
        scholtes_tot_it, 
        "$scholtes_status",
        "$(@sprintf "%.2e" ε)",
        "$(@sprintf "%.2e" τ)",
        "$(@sprintf "%.2e" cons_vio_inf)",
        "$(@sprintf "%.2e" scholtes_V)")
println("===========================================================================================================================")

    return (;
        tot_it = scholtes_tot_it, 
        f_opt = f_opt,           
        cons_vio = scholtes_V,
        cons_vio_inf = cons_vio_inf,
        time = scholtes_elapsed_time,
        status = scholtes_status,
        ε = ε,      
        τ = τ, 
        x = copy(x)
    )
end

end # nlp solvers module