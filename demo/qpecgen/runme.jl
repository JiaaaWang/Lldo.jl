"""

QPECgen benchmark:

	min  0.5*[x_var;y_var]^T*P*[x_var;y_var] + [c;d]^T*[x_var;y_var]
	s.t. A*[x_var;y_var] + a <= 0
         0<= (N*x_var + M*y_var + q) _|_ (y_var) >= 0

To reproduce the results in this paper, use the parameter settings reported therein and generate the following problems:

    "10_10_1", "10_10_2", "10_10_3", "10_10_4", "10_10_5",
    "20_20_1", "20_20_2", "20_20_3", "20_20_4", "20_20_5",
    "30_30_1", "30_30_2", "30_30_3", "30_30_4", "30_30_5",
    "40_40_1", "40_40_2", "40_40_3", "40_40_4", "40_40_5",
    "50_50_1", "50_50_2", "50_50_3", "50_50_4", "50_50_5",
    "100_100_1", "100_100_2", "100_100_3", "100_100_4", "100_100_5",
    "200_200_1", "200_200_2", "200_200_3", "200_200_4", "200_200_5",
    "500_500_1", "500_500_2", "500_500_3", "500_500_4", "500_500_5",
    "1000_1000_1", "1000_1000_2", "1000_1000_3", "1000_1000_4", "1000_1000_5"
            
"""

using Printf
using LinearAlgebra
using MAT
using DataFrames
using CSV
using JLD2

include("../../src/Lldo.jl") 
using .Lldo

include("../../model/mod_qpecgen.jl") 
using .mod_qpecgen

# problem info.
prob_type = "cc"
prob_name = "10_10_1"
current_dir = @__DIR__
mat_path = joinpath(@__DIR__, "data/$(prob_name).mat")
mat_data = matread(mat_path)

# load problem data
data_a = mat_data["a"] 
data_A = mat_data["A"]
data_c = mat_data["c"]
data_d = mat_data["d"]
data_M = mat_data["M"]
data_N = mat_data["N"]
data_P = mat_data["P"]
data_q = mat_data["q"]
xgen = mat_data["xgen"]
ygen = mat_data["ygen"]

r = size(data_N, 1)     # number of complementarity constraints
m = length(data_a)      # number of general constraints
n = size(data_P, 1)     # variable dimension

# construct inputs for Lldo
f = ScalarFunction(:f, create_f_func(data_P, data_c, data_d), create_f_grad(data_P, data_c, data_d))

F = Vector{VectorFunction}()
y0 = Vector{Vector{Float64}}() 

if prob_type == "vcc"

    for i in 1 : r
        F_i = VectorFunction(:F, create_F_func(data_M, data_N, data_q, i), create_F_jaco(data_M, data_N, data_q, i))
        push!(F, F_i)
        push!(y0, zeros(2))
    end

elseif prob_type == "cc"

    G_CC = VectorFunction(:G, create_G_func(data_M, data_N, data_q), create_G_jaco(data_M, data_N, data_q))
    push!(F, G_CC)
    push!(y0, zeros(r))

    H_CC = VectorFunction(:H, create_H_func(data_M, data_N, data_q), create_H_jaco(data_M, data_N, data_q))
    push!(F, H_CC)
    push!(y0, zeros(r))

end

A = VectorFunction(:A, create_A_func(data_A, data_a), create_A_jaco(data_A, data_a))
C = BoxSet(:ineq, fill(-Inf, m), fill(0.0, m))
push!(y0, zeros(m))
x0 = zeros(n)
tol = 1e-4

# start test 
run(`clear`)

# algencan
result_algencan = direct_algencan(
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

# classical ALM
opt_class_alm = ClassALMOptions{Float64}(
    tol_prim = tol,
    tol_dual = tol,
    ε0 = tol,
    κ_ε = 1.0,
    prob_type = prob_type,
    verbose = false
)

result_class_alm = class_alm(
    f,
    F,
    A,
    C,
    x0,
    y0,
    r;
    options = opt_class_alm
)

# smoothed homotopy 
opt_smooth_homotopy = SmoothHomotopyOptions{Float64}(
    tol_prim = tol,
    tol_dual = tol,
    ε0 = tol,
    κ_ε = 1.0,
    prob_type = prob_type,
    verbose = false
)

result_smooth_homotopy = smooth_homotopy(
    f,
    F,
    A,
    C,
    x0,
    r;
    options = opt_smooth_homotopy
)

# smoothed ALM 
opt_smooth_alm = SmoothALMOptions{Float64}(
    tol_prim = tol,
    tol_dual = tol,
    ε0 = tol,
    κ_ε = 1.0,
    prob_type = prob_type,
    verbose = false
)

result_smooth_alm = smooth_alm(
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
        method = "Algencan",
        name = prob_name,
        obj = @sprintf("%.2f", result_algencan.f_opt),
        vio = @sprintf("%.2e", result_algencan.cons_vio_inf),
        outer_iters = missing,
        inner_iters = missing,
        runtime = @sprintf("%.2f", result_algencan.time),
        status = string(result_algencan.status)
    ),
)

push!(
    data_store,
    (
        method = "Classical ALM",
        name = prob_name,
        obj = @sprintf("%.2f", result_class_alm.f_opt),
        vio = @sprintf("%.2e", result_class_alm.cons_vio),
        outer_iters = result_class_alm.tot_it,
        inner_iters = result_class_alm.tot_inner_it,
        runtime = @sprintf("%.2f", result_class_alm.time),
        status = string(result_class_alm.status)
    ),
)

push!(
    data_store,
    (
        method = "Smoothed homotopy",
        name = prob_name,
        obj = @sprintf("%.2f", result_smooth_homotopy.f_opt),
        vio = @sprintf("%.2e", result_smooth_homotopy.cons_vio),
        outer_iters = result_smooth_homotopy.tot_it,
        inner_iters = result_smooth_homotopy.tot_inner_it,
        runtime = @sprintf("%.2f", result_smooth_homotopy.time),
        status = string(result_smooth_homotopy.status)
    ),
)

push!(
    data_store,
    (
        method = "Smoothed ALM",
        name = prob_name,
        obj = @sprintf("%.2f", result_smooth_alm.f_opt),
        vio = @sprintf("%.2e", result_smooth_alm.cons_vio),
        outer_iters = result_smooth_alm.tot_it,
        inner_iters = result_smooth_alm.tot_inner_it,
        runtime = @sprintf("%.2f", result_smooth_alm.time),
        status = string(result_smooth_alm.status)
    ),
)

result_dir = joinpath(@__DIR__, "result")
mkpath(result_dir)

filename = joinpath(result_dir, "$(prob_name).jld2")
@save filename result_algencan result_class_alm result_smooth_homotopy result_smooth_alm

filepath = joinpath(@__DIR__, "result", prob_name)
CSV.write(filepath * ".csv", data_store, header = true)