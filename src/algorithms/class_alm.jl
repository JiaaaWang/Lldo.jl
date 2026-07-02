""" 

Classical ALM :
    
    Subproblem:
        min f(x) + dist(A(x) + λ * yhat_C, C)^2 / (2 * λ)
                 + ∑ ||F_j(x) - w_j + λ * yhat_j||^2 / (2 * λ)

        s.t. w_j ∈ D_j, j = 1,…,r

    For VCCs:
        constraints: F := { F_1,…, F_r }
        auxiliary variables: w := { w_1,…, w_r }
        multiplier: yhat := { yhat_1,…, yhat_r, yhat_C }
        residuals: v := ( v_1,…, v_r, v_C )
        λ := ( λ_1,…, λ_r, λ_C )

    For CCs:
        constraints: F := { G, H }
        auxiliary variables: w := { w_G, w_H }
        multiplier: yhat := { yhat_G, yhat_H, yhat_C }
        λ := ( λ_1,…, λ_r, λ_C )

"""

module ClassALM

using LinearAlgebra
using Printf

using ..BoxSetUtils
using ..DisjunctiveSetUtils
using ..VCCsUtils
using ..CCsUtils
using ..FunctionStruct

export ClassALMOptions, class_alm

# helper functions of tailored CCs 
function solve_subproblem_CC(
    f::ScalarFunction,
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x0::AbstractVector,
    w0::Vector{<:AbstractVector},
    yhat::Vector{<:AbstractVector},
    λ::Vector{<:Real},
    nu::Int; # iteration index for the ALM outer loop; 
    tau::Real = 5.0,
    sigma::Real = 1e-4,
    m::Int = 10, # horizon for nonmonotone Armijo rule
    gamma_min::Real = 1.0,
    gamma_max::Real = 1e20,
    gamma_0::Real = 1.0,
    subsolver_tol::Real = 1e-4,
    subsolver_maxit::Int = 10000
)

    value_subproblem(x, w) = begin
        value_obj = 0.0
        total_G = 0.0
        total_H = 0.0

        value_f, _ = f(x)
        
        # about CCs
        G_CC = F[1]
        H_CC = F[2]
        yhat_G = yhat[1]
        yhat_H = yhat[2]
        w_G = w[1]
        w_H = w[2]

        λ_CC = λ[1:end-1]

        value_G_CC, _ = G_CC(x)
        value_H_CC, _ = H_CC(x)
        
        total_G = norm(value_G_CC .- w_G .+ λ_CC .* yhat_G, 2)^2 / (2 * λ_CC[1])
        total_H = norm(value_H_CC .- w_H .+ λ_CC .* yhat_H, 2)^2 / (2 * λ_CC[1])

        # about A and C
        λ_C = λ[end]
        yhat_C = yhat[3]
        value_A, _ = A(x)
        value_A_C = dist_C(value_A .+ λ_C .* yhat_C, C)^2 / (2 * λ_C)
        
        value_obj = value_f + total_G + total_H + value_A_C
        return value_obj
    end

    grad_subproblem(x, w) = begin
        grad_x = zeros(length(x))
        grad_w = [zeros(length(w_i)) for w_i in w]

        _, grad_f = f(x)
        
        # about CCs
        G_CC = F[1]
        H_CC = F[2]
        yhat_G = yhat[1]
        yhat_H = yhat[2]
        w_G = w[1]
        w_H = w[2]

        λ_CC = λ[1:end-1]

        value_G_CC, jaco_G_CC = G_CC(x)
        value_H_CC, jaco_H_CC = H_CC(x)

        g1 = (value_G_CC .- w_G .+ λ_CC .* yhat_G) / λ_CC[1]
        g2 = (value_H_CC .- w_H .+ λ_CC .* yhat_H) / λ_CC[1]

        grad_w[1] = - g1
        grad_w[2] = - g2

        # about A and C
        λ_C = λ[end]
        yhat_C = yhat[3]
        value_A, jaco_A = A(x)
        grad_A_C = jaco_A' * (value_A .+ λ_C .* yhat_C - proj_C(value_A .+ λ_C .* yhat_C, C)) / λ_C
        
        grad_x .+= grad_f .+ jaco_G_CC' * g1 .+ jaco_H_CC' * g2 .+ grad_A_C

        return grad_x, grad_w
    end 

    # general spectral gradient method (k: outer index; l: inner index)
    inner_solved = false
    inner_can_stop = inner_solved
    
    k = 1
    gamma_k = gamma_0
    gamma_k_0 = gamma_k
    xk = copy(x0)
    wk = [copy(w_i) for w_i in w0]

    xk_old = copy(xk)
    wk_old = [copy(w_i) for w_i in wk]
    grad_xk_old = zeros(length(xk))
    grad_wk_old = [zeros(length(w_i)) for w_i in wk]
    
    obj_store = Float64[]
    push!(obj_store, value_subproblem(xk, wk))

    stop_cond = nothing
    sol_dist = nothing

    while !inner_can_stop

        step_found = false

        # gradient with respect to (x,w)
        grad_xk, grad_wk = grad_subproblem(xk, wk)

        # if not return, do PGM and linesearch
        m_k = min(m, k)

        if k > 1
            gamma_k = dot(grad_xk .- grad_xk_old, xk .- xk_old) 
                    + sum(dot(grad_wk[i] .- grad_wk_old[i], wk[i] .- wk_old[i]) for i in 1:2)
            gamma_k = gamma_k / (norm(xk .- xk_old)^2 + sum(norm(wk[i] .- wk_old[i])^2 for i in 1:2))
            gamma_k = min( max(gamma_k, gamma_min), gamma_max )
            gamma_k_0 = gamma_k
        end

        # try to find a step size in one spectral gradient step 
        gamma_kl = gamma_k
        l = 1

        while !step_found 

            # gradient step with respect to x
            xkl = xk .- (1 / gamma_kl) .* grad_xk

            # projected gradient step with respect to wG and wH
            wkl = [similar(w_i) for w_i in wk]
            wkl[1], wkl[2] = proj_CC(
                wk[1] .- (1 / gamma_kl) .* grad_wk[1], 
                wk[2] .- (1 / gamma_kl) .* grad_wk[2]
            )

            # check the return rule of outer loop
            grad_xkl, grad_wkl = grad_subproblem(xkl, wkl)
            
            stop_cond_x = norm( 
                gamma_kl .* (xk .- xkl) .+ grad_xkl .- grad_xk, 2
            )

            stop_cond_w = norm(
                [
                    norm(
                        gamma_kl .* (wk[i] .- wkl[i]) .+ grad_wkl[i] .- grad_wk[i], 2
                    ) for i in 1:2
                ]
            )
            
            stop_cond = norm([stop_cond_x, stop_cond_w])
            if stop_cond <= subsolver_tol
                xk = xkl
                wk = wkl
                inner_solved = true
                step_found = true
                break
            end

            # check nonmonotone Armijo rule
            obj_kl = value_subproblem(xkl, wkl)

            # rule 1
            sol_dist = sqrt(norm(xkl .- xk)^2 + sum(norm(wkl[i] .- wk[i])^2 for i in 1:2)) 
            neg_norm = - sol_dist^2 * gamma_kl / 2

            # rule 2
            inner_product = dot(grad_xk, xkl .- xk) 
                          + sum(dot(grad_wk[i], wkl[i] .- wk[i]) for i in 1:2)

            window_size = min(m_k, length(obj_store))
            start_idx = length(obj_store) - window_size + 1
            max_obj = maximum(obj_store[start_idx:end])
            
            # if obj_kl <= max_obj + sigma * inner_product # rule 2
            if obj_kl <= max_obj + sigma * neg_norm # rule 1

                gamma_k = gamma_kl
                xk_old = xk
                wk_old = wk

                grad_xk_old = grad_xk
                grad_wk_old = grad_wk

                xk = xkl
                wk = wkl
        
                push!(obj_store, obj_kl)
                step_found = true
            else
                gamma_kl = gamma_kl * tau
                l += 1
            end

        end

        k += 1
        inner_can_stop = inner_solved || k >= subsolver_maxit

    end

    x_opt = xk
    w_opt = wk
    num_iter = k
    return x_opt, w_opt, num_iter, inner_solved

end

function update_multiplier_CC(
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x_opt::AbstractVector,
    w_opt::Vector{<:AbstractVector},
    yhat::Vector{<:AbstractVector},
    λ::Vector{<:Real}
)

    y = Vector{Vector{Float64}}()  
    v = Vector{Vector{Float64}}()
    shift = Vector{Vector{Float64}}()

    # about CCs
    G_CC = F[1]
    H_CC = F[2]
    yhat_G = yhat[1]
    yhat_H = yhat[2]
    w_opt_G = w_opt[1]
    w_opt_H = w_opt[2]

    λ_CC = λ[1:end-1]

    value_G_CC, _ = G_CC(x_opt)
    value_H_CC, _ = H_CC(x_opt)

    v_G = zeros(length(value_G_CC))
    v_H = zeros(length(value_H_CC))

    shift_G =  (value_G_CC .- w_opt_G .- v_G) ./ λ_CC
    shift_H =  (value_H_CC .- w_opt_H .- v_H) ./ λ_CC

    y_G = yhat_G .+ shift_G
    y_H = yhat_H .+ shift_H

    push!(y, y_G)
    push!(y, y_H)

    push!(v, v_G)
    push!(v, v_H)

    push!(shift, shift_G)
    push!(shift, shift_H)

    # about A and C
    λ_C = λ[end]
    yhat_C = yhat[3]
    value_A, _ = A(x_opt)

    v_C = proj_C(value_A .+ λ_C .* yhat_C, C)
    shift_C = (value_A .- v_C) / λ_C
    y_C = yhat_C + shift_C

    push!(y, y_C)
    push!(v, v_C)
    push!(shift, shift_C)

    return y, v, shift
end

function calculate_residual_CC(
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x_opt::AbstractVector,
    w_opt::Vector{<:AbstractVector},
    yhat::Vector{<:AbstractVector},
    v::Vector{<:AbstractVector},
    λ::Vector{<:Real}
)

    # about CCs
    G_CC = F[1]
    H_CC = F[2]
    value_G_CC, _ = G_CC(x_opt)
    value_H_CC, _ = H_CC(x_opt)
    w_opt_G = w_opt[1]
    w_opt_H = w_opt[2]

    yhat_G = yhat[1]
    yhat_H = yhat[2]

    v_G = v[1]
    v_H = v[2]

    λ_CC = λ[1:end-1]

    V = sqrt.((value_G_CC .- w_opt_G .- v_G).^2 .+ (value_H_CC .- w_opt_H .- v_H).^2)
    cons_vio = abs.(min.(value_G_CC, value_H_CC))

    # about A and C
    value_A, _ = A(x_opt)
    v_C = v[end]
    push!(V, norm(value_A .- v_C))
    push!(cons_vio, dist_C_inf(value_A, C))

    # store inner values of G and H
    interim_compl = Vector{Vector{Float64}}()
    push!(interim_compl, value_G_CC)
    push!(interim_compl, value_H_CC)

    return cons_vio, V, interim_compl
end

# helper functions of general VCCs 
function solve_subproblem_VCC(
    f::ScalarFunction,
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x0::AbstractVector,
    w0::Vector{<:AbstractVector},
    yhat::Vector{<:AbstractVector},
    λ::Vector{<:Real},
    nu::Int; # iteration index for the ALM outer loop
    tau::Real = 5.0,
    sigma::Real = 1e-4,
    m::Int = 10, # horizon for nonmonotone Armijo rule
    gamma_min::Real = 1.0,
    gamma_max::Real = 1e20,
    gamma_0::Real = 1.0,
    subsolver_tol::Real = 1e-4,
    subsolver_maxit::Int = 10000
)

    value_subproblem(x, w) = begin
        value_obj = 0.0
        total_F_w = 0.0

        value_f, _ = f(x)

        for i in eachindex(F)
            F_i = F[i]
            yhat_i = yhat[i]
            λ_i = λ[i]
            w_i = w[i]
            
            value_F, _ = F_i(x)
            value_F_w = norm(value_F .- w_i .+ λ_i .* yhat_i, 2)^2 / (2 * λ_i)
            total_F_w += value_F_w
        end

        λ_C = λ[end]
        yhat_C = yhat[end]
        value_A, _ = A(x)
        value_A_C = dist_C(value_A .+ λ_C .* yhat_C, C)^2 / (2 * λ_C)

        value_obj = value_f + total_F_w + value_A_C
        
        return value_obj
    end

    grad_subproblem(x, w) = begin
        grad_x = zeros(length(x))
        grad_w = [zeros(length(w_i)) for w_i in w]

        _, grad_f = f(x)

        for i in eachindex(F)
            F_i = F[i]
            yhat_i = yhat[i]
            λ_i = λ[i]
            w_i = w[i]
            
            value_F, jaco_F = F_i(x)
            grad_F_w = jaco_F' * (value_F .+ λ_i .* yhat_i .- w_i) / λ_i
            grad_w[i] = - (value_F .+ λ_i .* yhat_i .- w_i) / λ_i
            grad_x .+= grad_F_w
        end

        λ_C = λ[end]
        yhat_C = yhat[end]
        value_A, jaco_A = A(x)
        grad_A_C = jaco_A' * (value_A .+ λ_C .* yhat_C - proj_C(value_A .+ λ_C .* yhat_C, C)) / λ_C
        grad_x .+= grad_f .+ grad_A_C

        return grad_x, grad_w
    end 
    
    # general spectral gradient method (k: outer index; l: inner index)
    inner_solved = false
    inner_can_stop = inner_solved
    
    k = 1
    gamma_k = gamma_0
    xk = copy(x0)
    wk = [copy(w_i) for w_i in w0]

    xk_old = copy(xk)
    wk_old = [copy(w_i) for w_i in wk]
    grad_xk_old = zeros(length(xk))
    grad_wk_old = [zeros(length(w_i)) for w_i in wk]

    obj_store = Float64[]
    push!(obj_store, value_subproblem(xk, wk))

    while !inner_can_stop

        # gradient with respect to (x,w)
        grad_xk, grad_wk = grad_subproblem(xk, wk)

        # if not return
        m_k = min(m, k)
        if k > 1
            gamma_k = dot(grad_xk .- grad_xk_old, xk .- xk_old) 
                    + sum(dot(grad_wk[i] .- grad_wk_old[i], wk[i] .- wk_old[i]) for i in eachindex(F))
            gamma_k = gamma_k / (norm(xk .- xk_old)^2 + sum(norm(wk[i] .- wk_old[i])^2 for i in eachindex(F)))
            gamma_k = min( max(gamma_k, gamma_min), gamma_max )
        end

        # try to find a step size in one spectral gradient step 
        gamma_kl = gamma_k
        step_found = false
        l = 1
        
        while !step_found 

            # gradient step with respect to x
            xkl = xk .- (1 / gamma_kl) .* grad_xk

            # projected gradient step with respect to wi
            wkl = [similar(w_i) for w_i in wk]
            for i in eachindex(F)
                wkl[i] = wk[i] .- (1 / gamma_kl) .* grad_wk[i]
                wkl[i] = proj_D(wkl[i])
            end

            # check the return rule of outer loop
            grad_xkl, grad_wkl = grad_subproblem(xkl, wkl)
            
            stop_cond_x = norm( 
                gamma_kl .* (xk .- xkl) .+ grad_xkl .- grad_xk, 2
            )

            stop_cond_w = norm(
                [
                    norm(
                        gamma_kl .* (wk[i] .- wkl[i]) .+ grad_wkl[i] .- grad_wk[i], 2
                    ) for i in eachindex(F)
                ]
            )
            
            if norm([stop_cond_x, stop_cond_w]) <= subsolver_tol
                xk = xkl
                wk = wkl
                inner_solved = true
                step_found = true
                break
            end

            # check nonmonotone Armijo rule
            obj_kl = value_subproblem(xkl, wkl)

            # rule 1
            neg_norm = - (norm(xkl .- xk)^2 + sum(norm(wkl[i] .- wk[i])^2 for i in eachindex(F))) * gamma_kl / 2

            # rule 2
            inner_product = dot(grad_xk, xkl .- xk) 
                        + sum(dot(grad_wk[i], wkl[i] .- wk[i]) for i in eachindex(F))

            window_size = min(m_k, length(obj_store))
            start_idx = length(obj_store) - window_size + 1
            max_obj = maximum(obj_store[start_idx:end])
            
            if obj_kl <= max_obj + sigma * neg_norm

                gamma_k = gamma_kl

                xk_old = xk
                wk_old = wk

                grad_xk_old = grad_xk
                grad_wk_old = grad_wk

                xk = xkl
                wk = wkl
                push!(obj_store, obj_kl)
                step_found = true
            else
                gamma_kl = gamma_kl * tau
                l += 1
            end

        end

        k += 1
        inner_can_stop = inner_solved || k >= subsolver_maxit

    end

    x_opt = xk
    w_opt = wk
    num_iter = k
    return x_opt, w_opt, num_iter, inner_solved
end

function update_multiplier_VCC(
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x_opt::AbstractVector,
    w_opt::Vector{<:AbstractVector},
    yhat::Vector{<:AbstractVector},
    λ::Vector{<:Real}
)

    y = Vector{Vector{Float64}}()  
    v = Vector{Vector{Float64}}()
    shift = Vector{Vector{Float64}}()

    # about F and w
    for i in eachindex(F)
        F_i = F[i]
        yhat_i = yhat[i]
        λ_i = λ[i]
        w_opt_i = w_opt[i]

        value_F, _ = F_i(x_opt)
        value_F_w = value_F .- w_opt_i 

        v_i = zeros(length(value_F))

        shift_i = (value_F_w .- v_i) / λ_i
        y_i = yhat_i .+ shift_i

        push!(v, v_i)
        push!(y, y_i)
        push!(shift, shift_i)
    end

    # about A and C
    λ_C = λ[end]
    yhat_C = yhat[end]
    value_A, _ = A(x_opt)

    v_C = proj_C(value_A .+ λ_C .* yhat_C, C)
    shift_C = (value_A .- v_C) / λ_C
    y_C = yhat_C .+ shift_C

    push!(v, v_C)
    push!(y, y_C)
    push!(shift, shift_C)

    return y, v, shift
end

function calculate_residual_VCC(
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x_opt::AbstractVector,
    w_opt::Vector{<:AbstractVector},
    yhat::Vector{<:AbstractVector},
    v::Vector{<:AbstractVector},
    λ::Vector{<:Real}
)

    V = Vector{Real}()
    cons_vio = Vector{Real}()
    interim_compl = Vector{Vector{Float64}}()

    # about F and w
    for i in eachindex(F)
        F_i = F[i]
        v_i = v[i]
        yhat_i = yhat[i]
        w_opt_i = w_opt[i]
                
        value_F, _ = F_i(x_opt)
        value_F_w = value_F .- w_opt_i 

        # residuals
        push!(V, norm(value_F_w .- v_i)) 
        push!(cons_vio, abs(minimum(value_F)))

        # # store inner values of VCCs
        push!(interim_compl, value_F)
    end

    # about A and C
    value_A, _ = A(x_opt)
    v_C = v[end]
    push!(V, norm(value_A .- v_C))
    push!(cons_vio, dist_C_inf(value_A, C))

    return cons_vio, V, interim_compl
end

function safeguarded_multiplier(
    y::Vector{<:AbstractVector};
    lower::Real = -1e20, 
    upper::Real = 1e20
)
    
    yhat = Vector{Vector{Float64}}()
    
    for i in eachindex(y) 
        y_i = y[i]
        yhat_i = min.(max.(y_i, lower), upper)
        
        push!(yhat, yhat_i)
    end

    return yhat
end

# options struct
Base.@kwdef mutable struct ClassALMOptions{T<:Real}

    λ0::Vector{T} = T[]
    ε0::T = one(T) * 0.1
    θ::T = one(T) * 0.9

    κ_λ::Vector{T} = T[]          # decrease of λ
    κ_ε::T = one(T) * 0.1         # decrease of subproblem tol. ε

    maxit::Int = 200
    subsolver_maxit::Int = 10000

    tol_prim::T = one(T) * 1e-6
    tol_dual::T = one(T) * 1e-6

    prob_type::String = "vcc"

    verbose::Bool = true

end

# classical alm main function
function class_alm(
    f::ScalarFunction,
    F::Vector{VectorFunction},
    A::VectorFunction,
    C::BoxSet,
    x0::AbstractVector,
    y0::Vector{<:AbstractVector},
    r::Int; # number of VCCs or CCs
    options::ClassALMOptions = ClassALMOptions{Float64}()
)

    start_time = time()

    # initialization
    opt = options
    T = eltype(x0)

    if isempty(opt.λ0)
        opt.λ0 = fill(one(eltype(x0)), r + 1)
    end

    if isempty(opt.κ_λ)
        opt.κ_λ = fill(T(0.9), r + 1)
    end

    tot_it = 0
    tot_inner_it = 0
    solved = false
    tired = tot_it >= opt.maxit

    x = copy(x0)
    y = copy(y0)
    v = copy(y0)
    λ = opt.λ0
    ε = opt.ε0
    
    # initial auxiliary variables 
    w = [zeros(length(yi)) for yi in y0[1:end-1]]

    norm_cons_vio = nothing
    V_old = nothing
    V = nothing
    tot_interim_compl = Vector{Vector{Vector{Float64}}}()

    # loop
    can_stop = solved || tired 
    if opt.verbose
        println("\n")
        println("Classical ALM progress begin:")
        println("Iter | Inner iters |Inner flag  | Inner tolerance | Objective  | Max violation  | Infea measure |")
        println("-----|-------------|------------|-----------------|------------|----------------|---------------|")
    end
    
    while !can_stop
        tot_it += 1
        yhat = safeguarded_multiplier(y)

        if opt.prob_type == "cc"

            # subproblem
            x_opt, w_opt, num_iter, inner_solved = solve_subproblem_CC(
                    f, F, A, C, x, w, yhat, λ, tot_it-1; 
                    subsolver_tol = ε, subsolver_maxit = opt.subsolver_maxit
            )

            # dual estimate update
            y, v, shift = update_multiplier_CC(F, A, C, x_opt, w_opt, yhat, λ)

            # residual
            cons_vio, V, interim_compl = calculate_residual_CC(F, A, C, x_opt, w_opt, yhat, v, λ)
            
        elseif opt.prob_type == "vcc"

            # subproblem
            x_opt, w_opt, num_iter, inner_solved = solve_subproblem_VCC(
                    f, F, A, C, x, w, yhat, λ, tot_it-1; 
                    subsolver_tol = ε, subsolver_maxit = opt.subsolver_maxit
            )

            # dual estimate update
            y, v, shift = update_multiplier_VCC(F, A, C, x_opt, w_opt, yhat, λ)

            # residual
            cons_vio, V, interim_compl = calculate_residual_VCC(F, A, C, x_opt, w_opt, yhat, v, λ)
            
        else
            error("Unknown problem type: '$(opt.prob_type)'. Use 'cc' or 'vcc'")
        end

        # record results
        tot_inner_it += num_iter
        sub_solved = inner_solved
        x .= x_opt
        w .= w_opt
        f_outer, _ = f(x)
        norm_cons_vio = maximum(cons_vio)
        V_old = V
        push!(tot_interim_compl, interim_compl)

        # print interim results
        if opt.verbose 
            @printf "%4d | %11d | %10s | %15.2e | %10.2e | %14.2e |%14.2e | \n" tot_it tot_inner_it sub_solved ε f_outer norm_cons_vio norm(V) 
        end

        # termination checks
        solved = (ε <= opt.tol_dual && sub_solved) && (norm(V) <= opt.tol_prim) 
        tired = tot_it >= opt.maxit
        can_stop = solved || tired

        # parameter update
        if !can_stop
            V_norm = norm(V)
            V_old_norm = norm(V_old)

            if V_norm > max(opt.θ * V_old_norm, opt.tol_prim)
                λ .*= opt.κ_λ
            end

            # update inner tolerance
            ε = max(opt.κ_ε * ε, opt.tol_dual)

        end

    end

    elapsed_time = time() - start_time
    f_opt, _ = f(x)

    # check the stationarity of returned point 
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
    cons_vio = norm_cons_vio

    println("\nClassical ALM progress summary:")    

println("========================================================================================================================================")
    @printf("%-12s %-15s %-12s %-12s %-18s %-15s %-15s %-15s %-15s\n", 
        "Time", "Objective", "Outer iter", "Inner iter", "Status", "Tolerance", "λ for A(x)∈C", "λ for F(x)∈D", "Max violation")
    println("----------------------------------------------------------------------------------------------------------------------------------------")
    @printf("%-12s %-15s %-12d %-12d %-18s %-15s %-15s %-15s %-15s\n",
        "$(round(elapsed_time, digits=2))s", 
        "$(@sprintf "%.2e" f_opt)", 
        tot_it, 
        tot_inner_it, 
        "$status",
        "$(@sprintf "%.2e" ε)",
        "$(@sprintf "%.2e" λ_C)",
        "$(@sprintf "%.2e" λ_F_min)", 
        "$(@sprintf "%.2e" cons_vio)")
println("========================================================================================================================================")

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
        x = copy(x),      
        w = copy(w),                 
        y = copy(y),
        tot_interim_compl = tot_interim_compl
    )
end

end # class alm module