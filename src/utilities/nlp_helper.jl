""" 
    helper and wrapper functions of Algencan, Gencan and Ipopt         
"""

module AlgencanHelper

using NLPModels
using NLPModelsAlgencan

using ..BoxSetUtils
using ..FunctionStruct

export algencan_solver, gencan_solver

    struct myNLPModel <: NLPModels.AbstractNLPModel{Float64, Vector{Float64}}
        meta::NLPModels.NLPModelMeta
        counters::NLPModels.Counters

        obj_f::Function
        grad_f::Function
        cons_f::Function
        jac_f::Function
    end

    function NLPModels.obj(
        nlp::myNLPModel,
        x::AbstractVector
    )
        nlp.counters.neval_obj += 1
        x_float = vec(Float64.(x))
        return Float64(nlp.obj_f(x_float))
    end

    function NLPModels.grad!(
        nlp::myNLPModel,
        x::AbstractVector,
        g::AbstractVector
    )
        nlp.counters.neval_grad += 1
        x_float = vec(Float64.(x))
        grad_val = nlp.grad_f(x_float)
        g .= vec(Float64.(grad_val))
        return g
    end

    function NLPModels.grad(
        nlp::myNLPModel,
        x::AbstractVector
    )
        nlp.counters.neval_grad += 1
        x_float = vec(Float64.(x))
        return vec(Float64.(nlp.grad_f(x_float)))
    end

    function NLPModels.cons!(
        nlp::myNLPModel,
        x::AbstractVector,
        c::AbstractVector
    )
        x_float = vec(Float64.(x))
        cons_val = nlp.cons_f(x_float)
        c .= vec(Float64.(cons_val))
        return c
    end

    function NLPModels.cons(
        nlp::myNLPModel,
        x::AbstractVector
    )
        x_float = vec(Float64.(x))
        return vec(Float64.(nlp.cons_f(x_float)))
    end

    function NLPModels.jac_structure(
        nlp::myNLPModel
    )
        m = nlp.meta.ncon
        n = nlp.meta.nvar

        rows = Int[]
        cols = Int[]

        # every entry of the Jacobian exists (dense)
        for j in 1:n
            for i in 1:m
                push!(rows, i)
                push!(cols, j)
            end
        end

        return rows, cols
    end

    function NLPModels.jac_coord!(
        nlp::myNLPModel,
        x::AbstractVector,
        vals::AbstractVector
    )
        x_float = vec(Float64.(x))
        J = Float64.(nlp.jac_f(x_float))
        vals .= vec(J)
        return vals
    end

    function NLPModels.hess_structure(nlp::myNLPModel)
        return Int[], Int[]
    end

    # callback for the Hessian of the Lagrangian
    function NLPModels.hess_coord(
        nlp::myNLPModel,
        x::AbstractVector,
        y::AbstractVector;
        obj_weight = 1.0
    )
        return Float64[]
    end
    
    function algencan_solver(
        f::ScalarFunction,
        A::VectorFunction,
        C::BoxSet,
        x0::AbstractVector;
        tol_dual::Real = 1e-6,
        tol_prim::Real = 1e-6
    )
        
        var_num = length(x0)

        function obj(x)
            value_f, _ = f(x)
            return value_f
        end

        function grad_obj(x)
            _, grad_f = f(x)
            return grad_f
        end

        function cons(x)
            value_A, _ = A(x)
            return value_A
        end

        function jaco_cons(x)
            _, jaco_A = A(x)
            return jaco_A
        end

        lvar = fill(-Inf, var_num)
        uvar = fill(Inf, var_num)

        lcon = C.lower
        ucon = C.upper

        # ==========================================
        # important setting in NLPModelsAlgencan.jl: 
        #       myevalhl = C_NULL
        #       myevalhlp = C_NULL
        #       coded[10] = UInt8(0)
        #       coded[11] = UInt8(0)
        # (run this pkg in developing mode for safe)
        # ==========================================
        meta = NLPModels.NLPModelMeta(
            var_num;
            x0 = vec(Float64.(x0)),
            lvar = lvar,
            uvar = uvar,
            ncon = length(lcon),
            lcon = lcon,
            ucon = ucon,
            nnzj = length(lcon) * var_num, # number of nonzeros in the constraint Jacobian (dense)
            nnzh = 0, # number of nonzeros in the Hessian of the Lagrangian
            minimize = true
        )

        nlp = myNLPModel(
            meta,
            NLPModels.Counters(),
            obj,
            grad_obj,
            cons,
            jaco_cons
        )

        result = redirect_stdout(devnull) do
            algencan(nlp; epsfeas = tol_prim, epsopt = tol_dual) 
        end

        x_opt = result.solution
        f_opt = result.objective
        algencan_status = result.status
        fun_evals = neval_obj(nlp)
        grad_evals = neval_grad(nlp)
        cons_vio = dist_C(cons(x_opt), C)
        cons_vio_inf = dist_C_inf(cons(x_opt), C)

        return (;
            fun_evals = fun_evals,
            grad_evals = grad_evals,  
            f_opt = f_opt,           
            cons_vio = cons_vio,
            cons_vio_inf = cons_vio_inf,
            status = algencan_status,
            x = copy(x_opt)
        )
    end

    # pass the unconstrained problem to Algencan
    function gencan_solver(
        f::ScalarFunction,
        x0::AbstractVector;
        ε::Real = 1e-6,
        tol_prim::Real = 1e-6
    )
        
        var_num = length(x0)

        function obj(x)
            value_f, _ = f(x)
            return value_f
        end

        function grad_obj(x)
            _, grad_f = f(x)
            return grad_f
        end

        function cons(x)
            return Float64[] # no constraints
        end

        function jaco_cons(x)
            return zeros(0, length(x)) # empty Jacobian
        end

        meta = NLPModels.NLPModelMeta(
            var_num;
            x0 = vec(Float64.(x0)),   
            lvar = fill(-Inf, var_num),
            uvar = fill(Inf, var_num),
            ncon = 0,
            lcon = Float64[],
            ucon = Float64[],
            nnzj = 0,
            nnzh = 0, 
            minimize = true
        )

        nlp = myNLPModel(
            meta,
            NLPModels.Counters(),
            obj,
            grad_obj,
            cons,
            jaco_cons
        ) 
        
        result = redirect_stdout(devnull) do
            algencan(nlp; epsfeas = tol_prim, epsopt = ε) 
        end

        x_opt = result.solution
        f_opt = result.objective
        gencan_status = result.status
        fun_evals = neval_obj(nlp)
        grad_evals = neval_grad(nlp)

        return (;
            fun_evals = fun_evals,
            grad_evals = grad_evals,  
            f_opt = f_opt,           
            status = gencan_status,
            x = copy(x_opt)
        )
    end

end # algencan module

module IpoptHelper

using Random
using Ipopt
using MathOptInterface

using ..BoxSetUtils
using ..FunctionStruct

const MOI = MathOptInterface

export ipopt_solver

    # identify the sparsity pattern manually
    function jacobian_sparsity(
        A::VectorFunction,
        var_num::Int;
        sample_num::Int = 30,
        jaco_tol::Real = 1e-12
    )
        S = Set{Tuple{Int,Int}}()
        
        for sample in 1 : sample_num

            x_test = randn(var_num)
            _, jaco_A = A(x_test)
            
            for i in axes(jaco_A, 1)

                for j in axes(jaco_A, 2)

                    if abs(jaco_A[i,j]) > jaco_tol
                        push!(S, (i, j))
                    end

                end

            end

        end
        
        return sort!(collect(S))
    end

    struct ipopt_evaluator <: MOI.AbstractNLPEvaluator
        objective::ScalarFunction
        constraint::VectorFunction
        sparsity::Vector{Tuple{Int,Int}}
    end

    MOI.features_available(d::ipopt_evaluator) = [:Grad, :Jac]

    function MOI.initialize(d::ipopt_evaluator, features)
        return
    end

    function MOI.eval_objective(
        d::ipopt_evaluator,
        x
    )
        f = d.objective
        value_f, _ = f(x)
        return value_f
    end

    function MOI.eval_objective_gradient(
        d::ipopt_evaluator,
        grad_f,
        x
    )
        f = d.objective
        _, grad = f(x)
        grad_f[:] = grad
        return
    end

    function MOI.eval_constraint(
        d::ipopt_evaluator,
        g,
        x
    )
        A = d.constraint
        value_A, _ = A(x)
        g[:] = value_A
        return
    end

    function MOI.eval_constraint_jacobian(
        d::ipopt_evaluator,
        J,
        x
    )
        A = d.constraint
        _, jaco_A = A(x)

        # fill J according to sparsity pattern
        for (idx, (i, j)) in enumerate(d.sparsity)
            J[idx] = jaco_A[i, j]
        end

        return
    end

    function MOI.jacobian_structure(
        d::ipopt_evaluator
    )
        return d.sparsity
    end

    function ipopt_solver(
        f::ScalarFunction,
        A::VectorFunction,
        C::BoxSet,
        x0::AbstractVector;
        tol_dual::Real = 1e-6,
        tol_prim::Real = 1e-6
    )

        model = Ipopt.Optimizer()

        MOI.set(model, MOI.Silent(), false)

        MOI.set(
            model,
            MOI.RawOptimizerAttribute(
                "hessian_approximation"
            ),
            "limited-memory"
        )

        var_num = length(x0)
        xvars = [MOI.add_variable(model) for _ in 1 : var_num]

        sparsity_A = jacobian_sparsity(A, var_num)

        evaluator = ipopt_evaluator(f, A, sparsity_A)

        C_lower = C.lower
        C_upper = C.upper
        bound = [MOI.NLPBoundsPair(C_lower[i], C_upper[i]) for i in eachindex(C_upper)]

        data_block = MOI.NLPBlockData(bound, evaluator, true)

        MOI.set(model, MOI.NLPBlock(), data_block)

        for i in 1 : var_num
            MOI.set(model, MOI.VariablePrimalStart(), xvars[i], x0[i])
        end

        MOI.set(model, MOI.RawOptimizerAttribute("dual_inf_tol"), tol_dual)
        MOI.set(model, MOI.RawOptimizerAttribute("constr_viol_tol"), tol_prim)

        MOI.optimize!(model)

        x_opt = [MOI.get(model, MOI.VariablePrimal(), xvars[i]) for i in 1 : var_num]
        f_opt = MOI.get(model, MOI.ObjectiveValue())

        value_A_opt, _ = A(x_opt)
        cons_vio = dist_C(value_A_opt, C)
        cons_vio_inf = dist_C_inf(value_A_opt, C)

        ipopt_status = MOI.get(model, MOI.TerminationStatus())
        ipopt_status = Symbol(string(ipopt_status))
        
        return (;
            cons_vio = cons_vio,
            cons_vio_inf = cons_vio_inf,
            status = ipopt_status,
            f_opt = f_opt,            
            x = copy(x_opt)
        )
    end

end # ipopt module