"""

MacMPEC benchmark:

	min  f(x)
	s.t. A(x) ∈ C
         0<= G(x) _|_ H(x) >= 0
         

To reproduce the results in this paper, use the following problem list:

    "bard1.nl", "bard1m.nl", "bard3.nl", "bard3m.nl", "bilevel1.nl", "bilevel1m.nl", 
    "bilevel2.nl", "bilevel2m.nl", "bilevel3.nl", "dempe.nl", "desilva.nl", "df1.nl",
    "ex9.1.1.nl", "ex9.1.2.nl", "ex9.1.3.nl", "ex9.1.4.nl", "ex9.1.5.nl", 
    "ex9.1.6.nl", "ex9.1.7.nl", "ex9.1.8.nl", "ex9.1.9.nl", "ex9.1.10.nl",
    "ex9.2.1.nl", "ex9.2.2.nl", "ex9.2.3.nl", "ex9.2.4.nl", "ex9.2.5.nl",
    "ex9.2.6.nl", "ex9.2.7.nl", "ex9.2.8.nl", "ex9.2.9.nl",
    "flp2.nl", "flp4-1.nl", "flp4-2.nl", "gauvin.nl", "hs044-i.nl", "liswet1-050.nl", 
    "gnash10.nl", "gnash11.nl", "gnash12.nl", "gnash13.nl", "gnash14.nl",
    "gnash15.nl", "gnash16.nl", "gnash17.nl", "gnash18.nl", "gnash19.nl",
    "incid-set1-8.nl", "incid-set1c-8.nl", "incid-set2-8.nl", "incid-set2c-8.nl", "incid-set-8.nl",
    "jr1.nl", "jr2.nl", "kth1.nl", "kth2.nl", "kth3.nl",
    "nash1a.nl", "nash1b.nl", "nash1c.nl", "nash1d.nl", "nash1e.nl",
    "outrata31.nl", "outrata32.nl", "outrata33.nl", "outrata34.nl",
    "pack-comp1-8.nl", "pack-comp1c-8.nl", "pack-comp2c-8.nl", "pack-comp-8.nl",
    "pack-rig1-8.nl", "pack-rig1c-8.nl", "pack-rig2-8.nl", 
    "pack-rig2c-8.nl", "pack-rig3-8.nl", "pack-rig3c-8.nl", 
    "portfl1.nl", "portfl2.nl", "portfl3.nl", "portfl4.nl", "portfl6.nl",
    "portfl-i-1.nl", "portfl-i-2.nl", "portfl-i-3.nl", "portfl-i-4.nl", "portfl-i-6.nl",
    "qpec1.nl", "qpec2.nl", "ralph1.nl", "ralph2.nl", "scale1.nl", "scale2.nl", 
    "scale3.nl", "scale4.nl", "scale5.nl", "scholtes1.nl", "scholtes2.nl", "scholtes3.nl", 
    "scholtes4.nl", "scholtes5.nl", "sl1.nl", "stackelberg1.nl"
            
"""

using Printf
using LinearAlgebra
using PyCall
using DataFrames
using CSV
using JLD2

include("../../src/Lldo.jl") 
using .Lldo

include("../../model/mod_macmpec.jl") 
using .mod_macmpec

pyjson = pyimport("json")
casadi_py = pyimport("casadi")

# problem info.
prob_type = "cc"
prob_name = "bard1.nl"
current_dir = @__DIR__
json_path = joinpath(@__DIR__, "data/$(prob_name).json")
json_text = read(json_path, String)
pydata = pyjson.loads(json_text)

# load problem data
f_fun_py = casadi_py.Function.deserialize(pydata["f_fun"])
g_fun_py = casadi_py.Function.deserialize(pydata["g_fun"])
G_fun_py = casadi_py.Function.deserialize(pydata["G_fun"])
H_fun_py = casadi_py.Function.deserialize(pydata["H_fun"])

w0_py = casadi_py.DM(pydata["w0"])
w0 = casadi_dm_to_julia(w0_py)
w0 = vec(w0)

lbw_py = casadi_py.DM(pydata["lbw"])
lbw = casadi_dm_to_julia(lbw_py)
lbw = vec(lbw)

ubw_py = casadi_py.DM(pydata["ubw"])
ubw = casadi_dm_to_julia(ubw_py)
ubw = vec(ubw)

lbg_py = casadi_py.DM(pydata["lbg"])
lbg = casadi_dm_to_julia(lbg_py)
lbg = vec(lbg)

ubg_py = casadi_py.DM(pydata["ubg"])
ubg = casadi_dm_to_julia(ubg_py)
ubg = vec(ubg)

size_G = get_dm_size(G_fun_py(w0_py))
size_g = get_dm_size(g_fun_py(w0_py))

n = get_dm_dimension(w0_py) # variable dimension
r = size_G[1]               # number of complementarity constraints
m = size_g[1]               # number of general constraints

bounded_var = findall(i -> (lbw[i] != -Inf) || (ubw[i] != Inf), eachindex(lbw))
n_bounded = length(bounded_var)
 
# construct inputs for Lldo
f = ScalarFunction(:f, create_f_func(f_fun_py), create_f_grad(f_fun_py))

F = Vector{VectorFunction}()
y0 = Vector{Vector{Float64}}() 

if prob_type == "vcc"

    F_func_combined = create_F_func(G_fun_py, H_fun_py)
    F_jaco_combined = create_F_jaco(G_fun_py, H_fun_py)

    for i in 1 : r

        function F_func_extract(x)
            F_mat = F_func_combined(x)
            return [F_mat[i, 1], F_mat[i, 2]]
        end

        function F_jaco_extract(x)
            J_full = F_jaco_combined(x)
            JG_i = J_full[i, :]         
            JH_i = J_full[r + i, :]    
            return [JG_i'; JH_i'] # the i or r+i row is stored as a column vector         
        end

        F_i = VectorFunction(:F, F_func_extract, F_jaco_extract)
        push!(F, F_i)
        push!(y0, zeros(2))

    end

elseif prob_type == "cc"

    G_CC = VectorFunction(:G, create_G_func(G_fun_py), create_G_jaco(G_fun_py))
    push!(F, G_CC)
    push!(y0, zeros(r))

    H_CC = VectorFunction(:H, create_H_func(H_fun_py), create_H_jaco(H_fun_py))
    push!(F, H_CC)
    push!(y0, zeros(r))

end

X = BoxSet(:bound, lbw, ubw)
A = VectorFunction(:A, create_A_func(g_fun_py, X), create_A_jaco(g_fun_py, X))
C = BoxSet(:ineq, vcat(lbg, lbw[bounded_var]), vcat(ubg, ubw[bounded_var]))

push!(y0, zeros(m+n_bounded))
x0 = zeros(n)
x0 .= w0
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
    m + n_bounded, # dimension of general constraints and bounded variables
    r; # number of VCCs or CCs
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