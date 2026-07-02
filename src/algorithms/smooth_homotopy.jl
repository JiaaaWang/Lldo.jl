""" 

Smoothed Homotopy:

    Subproblem:
        min f(x) + dist(A(x), C)^2 / (2 * λ)
                 + ∑ rβ_j(F_j(x)) / λ

    For VCCs:
        constraints: F := { F_1,…, F_r }
        residuals: v := { v_1,…, v_r, v_C}
        λ := ( λ_1,…, λ_r, λ_C )
        μ := ( μ_1,…, μ_r )

    For CCs:
        constraints: F := { G, H }
        λ := ( λ_1,…, λ_r, λ_C )
        μ := ( μ_1,…, μ_r )

"""

module SmoothHomotopy

using LinearAlgebra
using Optim
using Printf

using ..BoxSetUtils
using ..DisjunctiveSetUtils
using ..VCCsUtils
using ..CCsUtils
using ..FunctionStruct

export SmoothHomotopyOptions, smooth_homotopy

# helper functions of tailored CCs 
function solve_subproblem_CC(
    f::ScalarFunction,
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x0::AbstractVector,
    λ::Vector{<:Real},
    μ::Vector{<:Real}; 
    subsolver_tol::Real = 1e-6,
    subsolver_maxit::Int = 10000
)

    value_subproblem(x) = begin
        value_f, _ = f(x)
        
        # double envelope
        G_CC = F[1]
        H_CC = F[2]

        λ_CC = λ[1:end-1]
        μ_CC = μ

        value_G_CC, _ = G_CC(x)
        value_H_CC, _ = H_CC(x)

        value_LL = LL_double_env_compl(
            value_G_CC, 
            value_H_CC, 
            λ_CC, μ_CC
        )

        # Moreau envelope
        λ_C = λ[end]
        value_A, _ = A(x)
        value_Moreau = dist_C(value_A, C)^2 / (2 * λ_C)
        
        return value_f + sum(value_LL) + value_Moreau
    end

    grad_subproblem!(G, x) = begin
        _, grad_f = f(x)

        grad_LL = zeros(length(x))
        
        # double envelope
        G_CC = F[1]
        H_CC = F[2]

        λ_CC = λ[1:end-1]
        μ_CC = μ

        value_G_CC, jaco_G_CC = G_CC(x)
        value_H_CC, jaco_H_CC = H_CC(x)

        g1, g2 = grad_LL_double_env_compl(
            value_G_CC, 
            value_H_CC, 
            λ_CC, μ_CC
        )

        grad_LL = jaco_G_CC' * g1 .+ jaco_H_CC' * g2

        # Moreau envelope
        λ_C = λ[end]
        value_A, jaco_A = A(x)
        grad_Moreau = jaco_A' * (value_A - proj_C(value_A, C)) / λ_C
        
        G[:] = grad_f .+ grad_LL .+ grad_Moreau
    end 

    subproblem = OnceDifferentiable(value_subproblem, 
                                    grad_subproblem!, 
                                    x0)

    subproblem_result = optimize(
        subproblem, 
        x0, 
        LBFGS(),
        Optim.Options(g_tol = subsolver_tol, iterations = subsolver_maxit)
    )

    return (;
        x_opt = Optim.minimizer(subproblem_result), 
        num_inner_iter = subproblem_result.iterations, 
        success_flag = Optim.converged(subproblem_result)
    )
end

function calculate_residual_CC(
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x_opt::AbstractVector,
    λ::Vector{<:Real},
    μ::Vector{<:Real},
    β_underline::Real
)

    y = Vector{Vector{Float64}}()

    # double envelope
    G_CC = F[1]
    H_CC = F[2]
    value_G_CC, jaco_G_CC = G_CC(x_opt)
    value_H_CC, jaco_H_CC = H_CC(x_opt)

    λ_CC = λ[1:end-1]
    μ_CC = μ
    β =  μ_CC ./ λ_CC

    cons_vio = abs.(min.(value_G_CC, value_H_CC))

    index_Tb = (value_G_CC .> 0) .& 
            (value_H_CC .< (λ_CC ./ (λ_CC .- μ_CC)) .* value_G_CC) .& 
            (value_H_CC .> ((λ_CC .- μ_CC) ./ λ_CC) .* value_G_CC)

    M_counter = count(index_Tb)
    M_indices = findall(index_Tb)

    if minimum(β) > β_underline
        v_G, v_H = proj_CC(value_G_CC, value_H_CC)
    else
        v_G, v_H = proj_CC_Tb(value_G_CC, value_H_CC, β)
    end

    V = sqrt.((value_G_CC.-v_G).^2 .+ (value_H_CC.-v_H).^2)

    y_G_CC, y_H_CC = grad_LL_double_env_compl(
            value_G_CC, 
            value_H_CC, 
            λ_CC, μ_CC
        )
    push!(y, y_G_CC)
    push!(y, y_H_CC)

    # Moreau envelope
    value_A, jaco_A = A(x_opt)
    v_C = proj_C(value_A, C)
    push!(V, norm(value_A .- v_C))
    push!(cons_vio, dist_C_inf(value_A, C))

    λ_C = λ[end]
    y_C = (value_A - proj_C(value_A, C)) / λ_C
    push!(y, y_C)

    # store inner values of G and H
    interim_compl = Vector{Vector{Float64}}()
    push!(interim_compl, value_G_CC)
    push!(interim_compl, value_H_CC)

    return cons_vio, V, M_counter, M_indices, interim_compl, y
end

# helper functions of general VCCs 
function solve_subproblem_VCC(
    f::ScalarFunction,
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x0::AbstractVector,
    λ::Vector{<:Real},
    μ::Vector{<:Real}; 
    subsolver_tol::Real = 1e-6,
    subsolver_maxit::Int = 10000
)

    value_subproblem(x) = begin
        value_f, _ = f(x)
        total_LL = 0.0
        
        # double envelope
        for i in eachindex(F)
            F_i = F[i]
            λ_i = λ[i]
            μ_i = μ[i]
            
            value_F, _ = F_i(x)
            value_LL, _ = LL_env_grad(value_F, λ_i, μ_i)
            total_LL += value_LL
        end

        # Moreau envelope
        λ_C = λ[end]
        value_A, _ = A(x)
        value_Moreau = dist_C(value_A, C)^2 / (2 * λ_C)
        
        return value_f + total_LL + value_Moreau
    end

    grad_subproblem!(G, x) = begin
        _, grad_f = f(x)

        total_grad = zeros(length(x))
        
        # double envelope
        for i in eachindex(F)
            F_i = F[i]
            λ_i = λ[i]
            μ_i = μ[i]
            
            value_F, jaco_F = F_i(x)
            _, grad_LL = LL_env_grad(value_F, λ_i, μ_i)
            total_grad .+= jaco_F' * grad_LL
        end

        # Moreau envelope
        λ_C = λ[end]
        value_A, jaco_A = A(x)
        grad_Moreau = jaco_A' * (value_A .- proj_C(value_A, C)) / λ_C
        
        G[:] = grad_f .+ total_grad .+ grad_Moreau
    end 

    subproblem_VCC = OnceDifferentiable(value_subproblem, 
                                        grad_subproblem!, 
                                        x0)

    subproblem_result = optimize(
        subproblem_VCC, 
        x0, 
        LBFGS(),
        Optim.Options(g_tol = subsolver_tol, iterations = subsolver_maxit)
    )

    return (;
        x_opt = Optim.minimizer(subproblem_result), 
        num_inner_iter = subproblem_result.iterations, 
        success_flag = Optim.converged(subproblem_result)
    )
end

function calculate_residual_VCC(
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x_opt::AbstractVector,
    λ::Vector{<:Real},
    μ::Vector{<:Real},
    β_underline::Real
)

    M_counter = 0
    M_indices = Int[]
    cons_vio = Vector{Real}()
    V = Vector{Real}()
    interim_compl = Vector{Vector{Float64}}()
    y = Vector{Vector{Float64}}()

    # double envelope
    for i in eachindex(F)
        F_i = F[i]
                
        value_F, jaco_F = F_i(x_opt)
        
        β_i =  μ[i] / λ[i]

        # check if F_i belongs to the Tβ regions
        M_temp = find_M_practice(value_F, λ[i], μ[i])

        if M_temp !== nothing          # all elements are positive
            if length(M_temp) != 1     # belong to Tβ
                M_counter += 1         # counter increase
                push!(M_indices, i)    # record the index of F_i
            end
        end

        if β_i > β_underline
            v_i = proj_D(value_F)
        else
            if M_temp !== nothing      # all elements are positive
                if length(M_temp) != 1     # belong to Tβ
                    v_i = copy(value_F)
                    v_i[M_temp] .= 0.0
                else
                    v_i = proj_D(value_F)
                end
            else
                v_i = proj_D(value_F)
            end
        end
        
        # residuals
        push!(cons_vio, abs(minimum(value_F)))
        push!(V, norm(value_F .- v_i))
        
        # store inner values of VCCs
        push!(interim_compl, value_F)

        _, y_i = LL_env_grad(value_F, λ[i], μ[i])
        push!(y, y_i)
    end

    # Moreau envelope
    value_A, jaco_A = A(x_opt)
    v_C = proj_C(value_A, C)
    push!(cons_vio, dist_C_inf(value_A, C))
    push!(V, norm(value_A .- v_C))

    y_C = (value_A .- proj_C(value_A, C)) / λ[end]
    push!(y, y_C)

    return cons_vio, V, M_counter, M_indices, interim_compl, y
end

# options struct
Base.@kwdef mutable struct SmoothHomotopyOptions{T<:Real}

    λ0::Vector{T} = T[]
    μ0::Vector{T} = T[]
    β0::Vector{T} = T[]

    ε0::T = one(T) * 0.1

    κ_λ::Vector{T} = T[]          # decrease of λ
    κ_β::T = one(T) * 0.9         # decrease of β
    κ_ε::T = one(T) * 0.1         # decrease of subproblem tol. ε

    β_low::T = one(T) * 1e-3      # lower bound of β
    β_underline::T = one(T) * 1.0 # Tb projection 

    maxit::Int = 200
    subsolver_maxit::Int = 10000

    tol_prim::T = one(T) * 1e-6
    tol_dual::T = one(T) * 1e-6

    prob_type::String = "vcc"

    verbose::Bool = true

end

# homotopy main function
function smooth_homotopy(
    f::ScalarFunction,
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x0::AbstractVector,
    r::Int; # number of VCCs or CCs;
    options::SmoothHomotopyOptions = SmoothHomotopyOptions{Float64}()
)

    start_time = time()

    # initialization
    opt = options
    T = eltype(x0)

    if isempty(opt.λ0)
        opt.λ0 = fill(one(eltype(x0)), r + 1)
    end

    if isempty(opt.μ0)
        opt.μ0 = fill(T(0.9), r)
    end

    if isempty(opt.β0)
        opt.β0 = fill(T(0.9), r)
    end

    if isempty(opt.κ_λ)
        opt.κ_λ = fill(T(0.9), r + 1)
    end

    tot_it = 0
    tot_inner_it = 0
    solved = false
    tired = tot_it >= opt.maxit

    x = copy(x0)
    λ = opt.λ0
    μ = opt.μ0
    β = opt.β0
    ε = opt.ε0

    norm_cons_vio = nothing
    tot_interim_compl = Vector{Vector{Vector{Float64}}}()
    catch_M_indices = nothing
    y = Vector{Vector{Float64}}()
    V = nothing

    # loop
    can_stop = solved || tired
    if opt.verbose
        println("\n")
        println("Smoothed homotopy progress begin:")
        println("Iter | Inner iters |Inner flag  | Inner tolerance | Objective  | Max violation  | Infea measure | Need smoothing|")
        println("-----|-------------|------------|-----------------|------------|----------------|---------------|---------------")
    end

    while !can_stop
        tot_it += 1

        if opt.prob_type == "cc"

            # subproblem
            inner_result = solve_subproblem_CC(
                f, F, A, C, x, λ, μ; 
                subsolver_tol = ε, subsolver_maxit = opt.subsolver_maxit
            )

            # residual and problematic area counter
            cons_vio, V, M_counter, M_indices, interim_compl, y = calculate_residual_CC(F, A, C, inner_result.x_opt, λ, μ, opt.β_underline)

        elseif opt.prob_type == "vcc"

            # subproblem
            inner_result = solve_subproblem_VCC(
                f, F, A, C, x, λ, μ; 
                subsolver_tol = ε, subsolver_maxit = opt.subsolver_maxit
            )

            # residual and problematic area counter
            cons_vio, V, M_counter, M_indices, interim_compl, y = calculate_residual_VCC(F, A, C, inner_result.x_opt, λ, μ, opt.β_underline)

        else
            error("Unknown problem type: '$(opt.prob_type)'. Use 'cc' or 'vcc'")
        end
        
        # record results
        tot_inner_it += inner_result.num_inner_iter
        sub_solved = inner_result.success_flag
        x .= inner_result.x_opt
        f_outer, _ = f(x)
        norm_cons_vio = maximum(cons_vio)
        push!(tot_interim_compl, interim_compl)
        catch_M_indices = M_indices

        # print interim results
        if opt.verbose 
            @printf "%4d | %11d | %10s | %15.2e | %10.2e | %14.2e |%14.2e | %12d |\n" tot_it tot_inner_it sub_solved ε f_outer norm_cons_vio norm(V) M_counter
        end

        # termination checks
        solved = (ε <= opt.tol_dual && sub_solved) && (norm(V) <= opt.tol_prim)
        tired = tot_it >= opt.maxit
        can_stop = solved || tired

        # parameter update
        if !can_stop
            # update penalty parameters
            λ .*= opt.κ_λ

            # shrink Tβ regions if some points belong there
            for i in 1 : r
                if i in M_indices
                    β[i] = max(β[i] * opt.κ_β, opt.β_low)
                end
            end

            # update inner tolerance
            ε = max(opt.κ_ε * ε, opt.tol_dual)
        end

        μ = λ[1:end-1] .* β
    end

    elapsed_time = time() - start_time
    f_opt, _ = f(x)

    # check the stationarity of returned point 
    Tβ_avoid = isempty(catch_M_indices) ? "Yes" : "No"

    status = 
    if solved 
        :sta
    elseif tired && (norm(V) <= opt.tol_prim)
        :max_iter_fea
    elseif tired && (norm(V) > opt.tol_prim)
        :max_iter_infea_sta
    else
        :unknown
    end

    λ_C = λ[end]     
    λ_F_min = minimum(λ[1:end-1])
    β_F_min = minimum(β)
    cons_vio = norm_cons_vio

    println("\nSmoothed homotopy progress summary:")
println("=====================================================================================================================================================")
    @printf("%-12s %-15s %-12s %-12s %-18s %-15s %-15s %-15s %-12s %-15s\n", 
        "Time", "Objective", "Outer iter", "Inner iter", "Status", "Tolerance", "λ for A(x)∈C", "λ for F(x)∈D", "β", "Max violation")
    println("-----------------------------------------------------------------------------------------------------------------------------------------------------")
    @printf("%-12s %-15s %-12d %-12d %-18s %-15s %-15s %-15s %-12s %-15s\n",
        "$(round(elapsed_time, digits=2))s", 
        "$(@sprintf "%.2e" f_opt)", 
        tot_it, 
        tot_inner_it, 
        "$status",
        "$(@sprintf "%.2e" ε)",
        "$(@sprintf "%.2e" λ_C)",
        "$(@sprintf "%.2e" λ_F_min)", 
        "$(@sprintf "%.2e" β_F_min)",
        "$(@sprintf "%.2e" cons_vio)")
println("=====================================================================================================================================================")

    return (;
        tot_it = tot_it,
        tot_inner_it = tot_inner_it,  
        f_opt = f_opt,           
        cons_vio = cons_vio,
        time = elapsed_time,
        status = status,
        ε = ε,
        λ_C = λ_C,       
        λ_F_min = λ_F_min,
        β = β,
        x = copy(x),
        y = copy(y), # double envelope gradient
        tot_interim_compl = tot_interim_compl,
        CC_in_Tβ = catch_M_indices,
        Tβ_avoid = Tβ_avoid
    )
end

end # smooth homotopy module