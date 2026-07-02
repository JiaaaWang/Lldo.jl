"""
    Example 6.4. in "The Augmented Lagrangian Method for Mathematical Programs with Vertical Complementarity Constraints Based on Inexact Scholtes Regularization"
"""

using LinearAlgebra
include("../../src/Lldo.jl") 
using .Lldo

function f_func(x)
    return 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
end

function f_grad(x)
    G = zeros(4)
    G[1] = -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1])  
    G[2] = 200 * (x[2] - x[1]^2)                          
    return G
end

function F_func_1(x)
    g11(x) = 0.75 + (x[1] - 1)^2 - (x[2] - 1)
    g12(x) = - 0.25 - (x[1] - 1)^2 - (x[2] - 1)
    g13(x) = 0.75 - (x[2] - 1.5)^2 - (x[1] - 1)
    g14(x) = 0.75 - (x[2] - 1.5)^2 + (x[1] - 1)
    
    return [ g11(x)+x[3], g12(x)+x[3], g13(x)+x[3], g14(x)+ x[3] ]
end

function F_jaco_1(x)
    J = zeros(4, 4)
    J[1,1] = 2*(x[1]-1); J[1,2] = -1; J[1,3] = 1
    J[2,1] = -2*(x[1]-1); J[2,2] = -1; J[2,3] = 1  
    J[3,1] = -1; J[3,2] = -2*(x[2]-1.5); J[3,3] = 1
    J[4,1] = 1; J[4,2] = -2*(x[2]-1.5); J[4,3] = 1
    return J
end

function F_func_2(x)
    g21(x) = 0.5 + (x[2] - 1) + 2 * (x[1] - 1)^2
    g22(x) = 1 - (x[2] - 1) - 2 * (x[1] - 1) - 4 * (x[1] - 1)^2  
    g23(x) = 1 - (x[2] - 1) + 2 * (x[1] - 1) - 4 * (x[1] - 1)^2 
    
    return [ g21(x)+x[4], g22(x)+x[4], g23(x)+x[4] ]
end

function F_jaco_2(x)
    J = zeros(3, 4)
    J[1,1] = 4*(x[1]-1); J[1,2] = 1; J[1,4] = 1
    J[2,1] = -2 - 8*(x[1]-1); J[2,2] = -1; J[2,4] = 1
    J[3,1] = 2 - 8*(x[1]-1); J[3,2] = -1; J[3,4] = 1
    return J
end

function A_func(x)
    return  x[3:4]
end

function A_jaco(x)
    J = zeros(2, 4)
    J[1,3] = 1
    J[2,4] = 1
    return J
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
C = BoxSet(:eq_ineq, fill(0.0, 2), fill(Inf, 2))
push!(y0, zeros(2))

x0 = zeros(4)
x0[1] = 2.0
x0[2] = 2.0

tol = 1e-4
r = 2
m = 2

# start test
run(`clear`)
prob_name = "mpvcc3"
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