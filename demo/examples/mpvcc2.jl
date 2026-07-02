"""
    Example 6.3. in "The Augmented Lagrangian Method for Mathematical Programs with Vertical Complementarity Constraints Based on Inexact Scholtes Regularization"
"""

using LinearAlgebra
include("../../src/Lldo.jl") 
using .Lldo

function f_func(x)
    return x[3]
end

function F_func_1(x)
    g11(x) = 1.5 + 0.5 * x[1]^2 - x[2]
    g12(x) = 0.5 + 0.5 * x[1]^2 - x[2] 
    g13(x) = - 1.5 + 0.5 * (x[2] - 1)^2 - x[1]
    g14(x) = - 1.5 + 0.5 * (x[2] - 1)^2 + x[1]
    
    return [ g11(x)+x[4], g12(x)+x[4], g13(x)+x[4], g14(x)+ x[4] ]
end

function F_func_2(x)
    g21(x) = - 5 + x[2] + x[1]^2
    g22(x) = 2 - x[2] + 2 * x[1] + 2 * x[1]^2  
    g23(x) = - 4 + x[2] - 2 * x[1] + 2 * x[1]^2
    
    return [ g21(x)+x[5], g22(x)+x[5], g23(x)+x[5] ]
end

function A_func(x)
    a1(x) = (x[1]-1)^2 + x[2]^2 - x[3]
    a2(x) = (x[1]-0.5)^2 + x[2]^2 - x[3]
    a3(x) = x[4]
    a4(x) = x[5]

    return  [a1(x), a2(x), a3(x), a4(x)]
end
       
f = ScalarFunction(:f, f_func, nothing)

F = Vector{VectorFunction}()
y0 = Vector{Vector{Float64}}() 

F_1 = VectorFunction(:F, F_func_1, nothing)
push!(F, F_1)
push!(y0, zeros(4))

F_2 = VectorFunction(:F, F_func_2, nothing)
push!(F, F_2)
push!(y0, zeros(3))

A = VectorFunction(:A, A_func, nothing)
C = BoxSet(:eq_ineq, [-Inf, -Inf, 0.0, 0.0], [0.0, 0.0, Inf, Inf])
push!(y0, zeros(4))

x0 = zeros(5)
x0[1] = 0.9
x0[2] = 0.9
tol = 1e-4
r = 2
m = 4

# start test
run(`clear`)
prob_name = "mpvcc2"
prob_type = "vcc"

println("="^30)
println("$prob_name: min-max-min problem")
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