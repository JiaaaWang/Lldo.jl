module mod_macmpec

using LinearAlgebra
using PyCall

const casadi_py = PyCall.pyimport("casadi")
const np = PyCall.pyimport("numpy")

export casadi_dm_to_julia,
       get_dm_dimension,
       get_dm_size,
       compute_jacobian,
       create_f_func,
       create_f_grad,
       create_F_func,
       create_F_jaco,
       create_A_func,
       create_A_jaco,
       create_G_func,
       create_G_jaco,
       create_H_func,
       create_H_jaco
    
    # useful functions to convert CasADi DM to Julia array 
    function casadi_dm_to_julia(dm_py)
        np_array = np.array(dm_py)
        return convert(Array{Float64}, np_array)
    end

    function get_dm_dimension(dm_py)
        np_array = np.array(dm_py)
        return length(np_array)
    end

    function get_dm_size(dm_py)
        np_array = np.array(dm_py)
        return size(np_array)
    end

    function compute_jacobian(func_py, x_py)
        y_py = func_py(x_py)
        y_jl = casadi_dm_to_julia(y_py)
        n_out = length(y_jl) # output dimension
        
        x_jl = casadi_dm_to_julia(x_py)
        n_in = length(x_jl) # input dimension
        
        if n_out == 0 # empty output - returning empty Jacobian matrix
            return zeros(0, n_in)
        end
        
        # create symbolic variables
        x_sym = casadi_py.MX.sym("x", n_in)
            
        # evaluate function symbolically
        f_sym = func_py(x_sym)
            
        # compute Jacobian matrix
        J_sym = casadi_py.jacobian(f_sym, x_sym)
            
        # create Jacobian function
        J_fun = casadi_py.Function("J", [x_sym], [J_sym])
            
        # evaluate at point
        J_py = J_fun(x_py)
        J_jl = casadi_dm_to_julia(J_py)
            
        return J_jl
    end

    # VCC constructors for f, f_grad, F, F_jaco, A, A_jaco 
    function create_f_func(f_fun_py)
        
        function f_func(x)

            x_py = casadi_py.DM(x)
            f_val_py = f_fun_py(x_py)
            f_val = casadi_dm_to_julia(f_val_py)

            return f_val[1]
        end
        
        return f_func
    end

    function create_f_grad(f_fun_py)
        
        function f_grad(x)

            x_py = casadi_py.DM(x)
            J_f = compute_jacobian(f_fun_py, x_py)
            J_f = vec(J_f')

            return J_f
        end
        
        return f_grad
    end

    function create_F_func(G_fun_py, H_fun_py)

        function F_func(x)

            x_py = casadi_py.DM(x)
            G_val_py = G_fun_py(x_py)
            H_val_py = H_fun_py(x_py)
            G_val = casadi_dm_to_julia(G_val_py)
            H_val = casadi_dm_to_julia(H_val_py)

            G_val = vec(G_val)
            H_val = vec(H_val)

            return [G_val H_val]
        end

        return F_func
    end

    function create_F_jaco(G_fun_py, H_fun_py)

        function F_jaco(x)

            x_py = casadi_py.DM(x)
            J_G = compute_jacobian(G_fun_py, x_py)
            J_H = compute_jacobian(H_fun_py, x_py)

            return [J_G; J_H]
        end

        return F_jaco
    end

    function create_A_func(g_fun_py, X)

        LB = X.lower
        UB = X.upper
        bounded_var = findall(i -> (LB[i] != -Inf) || (UB[i] != Inf), eachindex(LB))

        function A_func(x)

            x_py = casadi_py.DM(x)
            g_val_py = g_fun_py(x_py)
            g_val = casadi_dm_to_julia(g_val_py)
            g_val = vec(g_val)

            if isempty(bounded_var)
                return g_val
            else
                # extract bounded variables and stack with constraints
                bounded_vals = x[bounded_var]
                return [g_val; bounded_vals]
            end

        end

        return A_func
    end

    function create_A_jaco(g_fun_py, X)

        LB = X.lower
        UB = X.upper
        bounded_var = findall(i -> (LB[i] != -Inf) || (UB[i] != Inf), eachindex(LB))
        n_vars = length(LB)
        n_bounded = length(bounded_var)

        function A_jaco(x)

            x_py = casadi_py.DM(x)
            J_g = compute_jacobian(g_fun_py, x_py)

            if isempty(bounded_var)
                return J_g
            else
                # create selection matrix for bounded variables
                J_bounds = zeros(n_bounded, n_vars)

                for (i, idx) in enumerate(bounded_var)
                    J_bounds[i, idx] = 1.0
                end
                
                return [J_g; J_bounds]
            end

        end

        return A_jaco
    end

    # tailored for CCs
    function create_G_func(G_fun_py)

        function G_func(x)

            x_py = casadi_py.DM(x)
            G_val_py = G_fun_py(x_py)
            G_val = casadi_dm_to_julia(G_val_py)
            G_val = vec(G_val)

            return G_val
        end

        return G_func
    end

    function create_G_jaco(G_fun_py)

        function G_jaco(x)

            x_py = casadi_py.DM(x)
            J_G = compute_jacobian(G_fun_py, x_py)

            return J_G
        end

        return G_jaco
    end

    function create_H_func(H_fun_py)

        function H_func(x)

            x_py = casadi_py.DM(x)
            H_val_py = H_fun_py(x_py)
            H_val = casadi_dm_to_julia(H_val_py)
            H_val = vec(H_val)

            return H_val
        end

        return H_func
    end

    function create_H_jaco(H_fun_py)

        function H_jaco(x)

            x_py = casadi_py.DM(x)
            J_H = compute_jacobian(H_fun_py, x_py)

            return J_H
        end

        return H_jaco
    end

end # module