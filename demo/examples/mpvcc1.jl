"""
    Example 6.1. in "The Augmented Lagrangian Method for Mathematical Programs with Vertical Complementarity Constraints Based on Inexact Scholtes Regularization"
"""

using LinearAlgebra
include("../../src/Lldo.jl") 
using .Lldo

function f_func(x)
    return x[1]^2 + (x[5] - 1)^2 + (x[2] - x[6] + 1)^2 + exp(x[3]) + x[4]
end

function F_func_1(x)
    A1 = [2 1; 2 4]
    A2 = [1 3; 3 2]
    e = [1; 1]

    g1(x) = x[1] + x[5]
    g2(x) = (A1 * x[3:4] .- e)[1]
    g3(x) = (A2 * x[3:4] .- e)[1]
    
    return [ g1(x), g2(x), g3(x) ]
end

function F_func_2(x)
    A1 = [2 1; 2 4]
    A2 = [1 3; 3 2]
    e = [1; 1]

    g1(x) = x[2] + x[6]
    g2(x) = (A1 * x[3:4] .- e)[2]
    g3(x) = (A2 * x[3:4] .- e)[2]
    
    return [ g1(x), g2(x), g3(x) ]
end

function F_func_3(x)
    A3 = [1 2; 3 1]
    A4 = [5 1; 2 4]
    e = [1; 1]

    g1(x) = x[3]
    g2(x) = (A3' * (x[1:2] + x[5:6]) .- e)[1]
    g3(x) = (A4' * (x[1:2] + x[5:6]) .- e)[1]
    
    return [ g1(x), g2(x), g3(x) ]
end

function F_func_4(x)
    A3 = [1 2; 3 1]
    A4 = [5 1; 2 4]
    e = [1; 1]

    g1(x) = x[4]
    g2(x) = (A3' * (x[1:2] + x[5:6]) .- e)[2]
    g3(x) = (A4' * (x[1:2] + x[5:6]) .- e)[2]
    
    return [ g1(x), g2(x), g3(x) ]
end

function A_func(x)
    a1(x) = x[5]^2 + 2 * x[6] - 2
    a2(x) = x[5] + x[6] - 2

    return  [a1(x), a2(x)]
end

f = ScalarFunction(:f, f_func, nothing)

F = Vector{VectorFunction}()
y0 = Vector{Vector{Float64}}() 

F_1 = VectorFunction(:F, F_func_1, nothing)
push!(F, F_1)
push!(y0, zeros(3))

F_2 = VectorFunction(:F, F_func_2, nothing)
push!(F, F_2)
push!(y0, zeros(3))

F_3 = VectorFunction(:F, F_func_3, nothing)
push!(F, F_3)
push!(y0, zeros(3))

F_4 = VectorFunction(:F, F_func_4, nothing)
push!(F, F_4)
push!(y0, zeros(3))

A = VectorFunction(:A, A_func, nothing)
C = BoxSet(:eq_ineq, [0.0, -Inf], [0.0, 0])
push!(y0, zeros(2))
x0 = ones(6)
x0[4] = 0.0
tol = 1e-4
r = 4
m = 2

# start test
run(`clear`)
prob_name = "mpvcc1"
prob_type = "vcc"

println("="^30)
println("$prob_name: Stackelberg game")
println("="^30)

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