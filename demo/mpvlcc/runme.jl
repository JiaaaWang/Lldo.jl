"""

MPVLCC:

    min  f(x)
	s.t. w = Mz + q
         min{ z_i, w1_i, … , wl_i } = 0, for i = 1, … ,n

    For VCCs:
        x := (z, w1, … , wl) with z, w1, … , wl ∈ R^n
        w := (w1, … , wl)
        q := (q1, … , ql), q1, … , ql ∈ R^n
        M := (M1^T, … , Ml^T)^T, M1, … , Ml ∈ R^{nxn}

    For CC reformulation:
        x := (z, w1, … , wl, s1, … ,sl) 
        s := (s1, … ,sl) with s1, … ,sl ∈ R^n

    Generate problems as in 
    "Projected Splitting Methods for Vertical Linear Complementarity Problems", 
    but without imposing row diagonal dominance and positivity of the diagonal entries of Mi, 
    so that the solution need not be unique.

    To reproduce the results in this paper, use the following parameter settings:

        for l in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        for rand_seed in [1, 2, 3, 4, 5]

"""

using Printf
using LinearAlgebra
using Random
using DataFrames
using CSV
using JLD2

include("../../src/Lldo.jl") 
using .Lldo

include("../../model/mod_mpvlcc.jl") 
using .mod_mpvlcc

# generate random Mi matrix
function generate_Mi_matrix(n::Int)
    
    Mi = -1 .+ 2 .* rand(n, n)

    return Mi
end

# generate random M matrix
function generate_M_matrix(n::Int, l::Int)
    
    M = zeros(n * l, n)

    for i in 1 : l

        rows = (i - 1) * n + 1 : i * n
        M[rows, :] = generate_Mi_matrix(n)

    end

    return M
end

# generate random feasible solution
function generate_sol(n::Int, l::Int)
    
    x_opt = zeros(n + n * l)
    x_opt[1 : n] = rand(0 : 1, n) # z_opt

    for i in 1 : l

        rows = i * n + 1 : (i + 1) * n

        if i == 1 # w1_opt

            x_opt[rows] = ones(n)

            for j in i * n + ceil(Int, n / 2) : (i + 1) * n
                x_opt[j] = 1.0 - x_opt[j - i * n]
            end

        elseif i == 2 # w2_opt

            x_opt[rows] = ones(n)

            for j in i * n + 1 : i * n + floor(Int, n / 2) 
                x_opt[j] = 1.0 - x_opt[j - i * n]
            end

        else

            x_opt[rows] = rand(n)

        end

    end

    return x_opt
end

# start test 
run(`clear`)

n = 2   # number of VCCs
tol = 1e-4

for l in [100]

    for rand_seed in [1]

        prob_name = string(l, "_", rand_seed)

        # generate random data
        Random.seed!(rand_seed)

        M = generate_M_matrix(n, l)
        A = [M -Matrix(1.0I, n * l, n * l)]
        x_opt = generate_sol(n, l)
        q = - A * x_opt

        # VCC
        m = n * l
        var_num = n + n * l
        r = n
        prob_type = "vcc"

        f = ScalarFunction(:f, create_f_func(n, l), create_f_grad(n, l))
        F = Vector{VectorFunction}()
        y0 = Vector{Vector{Float64}}() 

        for i in 1 : r
            F_i = VectorFunction(:F, create_F_func(n, l, i), create_F_jaco(n, l, i))
            push!(F, F_i)
            push!(y0, zeros(l + 1))
        end

        A = VectorFunction(:A, create_A_func(n, l, M, q), create_A_jaco(n, l, M, q))
        C = BoxSet(:ineq, fill(0.0, m), fill(0.0, m))
        push!(y0, zeros(m))

        x0 = 20.0 .* rand(var_num) .- 10.0
        x0_vcc = similar(x0)
        x0_vcc .= x0

        # algencan (VCC)
        result_algencan_VCC = direct_algencan(
            f,
            F,
            A,
            C,
            x0,
            m, 
            r; 
            tol_prim = tol,
            tol_dual = tol,
            prob_type = prob_type
        )

        # smoothed ALM (VCC)
        opt_smooth_alm = SmoothALMOptions{Float64}(
            tol_prim = tol,
            tol_dual = tol,
            ε0 = tol,
            κ_ε = 1.0,
            prob_type = prob_type,
            verbose = false
        )

        result_alm_VCC = smooth_alm(
            f,
            F,
            A,
            C,
            x0,
            y0,
            r;
            options = opt_smooth_alm
        )

        # CC
        r = n * l
        var_num = n + n * l + n * l
        m = n * l + n
        prob_type = "cc"

        f = ScalarFunction(:f, create_f_CC_func(n, l), create_f_CC_grad(n, l))
        F = Vector{VectorFunction}()
        y0 = Vector{Vector{Float64}}() 

        G_CC = VectorFunction(:G, create_G_func(n ,l), create_G_jaco(n ,l))
        push!(F, G_CC)
        push!(y0, zeros(r))

        H_CC = VectorFunction(:H, create_H_func(n ,l), create_H_jaco(n ,l))
        push!(F, H_CC)
        push!(y0, zeros(r))

        A = VectorFunction(:A, create_A_CC_func(n, l, M, q), create_A_CC_jaco(n, l, M, q))
        C = BoxSet(:ineq, fill(0.0, m), fill(0.0, m))
        push!(y0, zeros(m))

        x0 = 20.0 .* rand(var_num) .- 10.0
        x0[1 : n + n * l] .= x0_vcc

        # algencan (CC)
        result_algencan_CC = direct_algencan(
            f,
            F,
            A,
            C,
            x0,
            m, 
            r; 
            tol_prim = tol,
            tol_dual = tol,
            prob_type = prob_type
        )

        # smoothed ALM (CC)
        opt_smooth_alm = SmoothALMOptions{Float64}(
            tol_prim = tol,
            tol_dual = tol,
            ε0 = tol,
            κ_ε = 1.0,
            prob_type = prob_type,
            verbose = false
        )

        result_alm_CC = smooth_alm(
            f,
            F,
            A,
            C,
            x0,
            y0,
            r;
            options = opt_smooth_alm
        )

        # save data
        data_store = DataFrame(
            method = Union{String, Missing}[],
            name = Union{String, Missing}[],
            obj = Union{String, Missing}[],  
            vio = Union{String, Missing}[], 
            outer_iters = Union{Int, Missing}[],
            inner_iters = Union{Int, Missing}[],
            runtime = Union{String, Missing}[],
            status = Union{String, Missing}[]
        )
        
        push!(
            data_store,
            (
                method = "Algencan (VCC)",
                name = prob_name,
                obj = @sprintf("%.2f", result_algencan_VCC.f_opt),
                vio = @sprintf("%.2e", result_algencan_VCC.cons_vio_inf),
                outer_iters = missing,
                inner_iters = missing,
                runtime = @sprintf("%.2f", result_algencan_VCC.time),
                status = string(result_algencan_VCC.status)
            ),
        )

        push!(
            data_store,
            (
                method = "Algencan (CC)",
                name = prob_name,
                obj = @sprintf("%.2f", result_algencan_CC.f_opt),
                vio = @sprintf("%.2e", result_algencan_CC.cons_vio_inf),
                outer_iters = missing,
                inner_iters = missing,
                runtime = @sprintf("%.2f", result_algencan_CC.time),
                status = string(result_algencan_CC.status)
            ),
        )

        push!(
            data_store,
            (
                method = "Smoothed ALM (VCC)",
                name = prob_name,
                obj = @sprintf("%.2f", result_alm_VCC.f_opt),
                vio = @sprintf("%.2e", result_alm_VCC.cons_vio),
                outer_iters = result_alm_VCC.tot_it,
                inner_iters = result_alm_VCC.tot_inner_it,
                runtime = @sprintf("%.2f", result_alm_VCC.time),
                status = string(result_alm_VCC.status)
            ),
        )

        push!(
            data_store,
            (
                method = "Smoothed ALM (CC)",
                name = prob_name,
                obj = @sprintf("%.2f", result_alm_CC.f_opt),
                vio = @sprintf("%.2e", result_alm_CC.cons_vio),
                outer_iters = result_alm_CC.tot_it,
                inner_iters = result_alm_CC.tot_inner_it,
                runtime = @sprintf("%.2f", result_alm_CC.time),
                status = string(result_alm_CC.status)
            ),
        )

        result_dir = joinpath(@__DIR__, "result")
        mkpath(result_dir)

        filename = joinpath(result_dir, "$(prob_name).jld2")
        @save filename result_alm_VCC result_alm_CC result_algencan_VCC result_algencan_CC 

        filepath = joinpath(@__DIR__, "result", prob_name)
        CSV.write(filepath * ".csv", data_store, header = true)

    end # seed

end # l