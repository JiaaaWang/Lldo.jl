module mod_qpecgen

using LinearAlgebra

export create_f_func,
       create_f_grad,
       create_F_func,
       create_F_jaco,
       create_A_func,
       create_A_jaco,
       create_G_func,
       create_G_jaco,
       create_H_func,
       create_H_jaco
    
    function create_f_func(P, c, d)
    
        function f_func(x)
            return 0.5 * x' * P * x + dot([c; d], x)
        end
        
        return f_func
    end

    function create_f_grad(P, c, d)
        
        function f_grad(x)
            return P * x + [c; d]
        end
        
        return f_grad
    end

    function create_F_func(M, N, q, i)
        length_x = size(N, 2)

        function F_func(x)
            # split x into [x_vars; y_vars]
            x_vars = x[1:length_x]
            y_vars = x[length_x+1:end]

            # g1: (y)_i >= 0  (complementarity part 1)
            g1 = y_vars[i]
            
            # g2: (Nx + My + q)_i >= 0  (complementarity part 2)  
            g2 = dot(N[i,:], x_vars) + dot(M[i,:], y_vars) + q[i]
            
            return [g1, g2]  
        end
        
        return F_func
    end
        
    function create_F_jaco(M, N, q, i)
        length_x = size(N, 2)
        length_y = size(M, 1)  # Note: should be size(M, 1) for rows
        total_vars = length_x + length_y
        
        function F_jaco(x)
            J = zeros(2, total_vars)  # 2 outputs × total variables
            
            # g1 = y_i
            J[1, length_x + i] = 1.0  # ∂g1/∂y_i = 1
            
            # g2 = N[i,:]*x + M[i,:]*y + q[i]
            # ∂g2/∂x = N[i,:]
            J[2, 1:length_x] = N[i, :]
            
            # ∂g2/∂y = M[i,:]  
            J[2, length_x+1:end] = M[i, :]
            
            return J
        end
        
        return F_jaco
    end
        
    function create_A_func(A, a)

        function A_func(x)
            return vec(A*x + a)
        end

        return A_func
    end

    function create_A_jaco(A, a)

        function A_jaco(x)
            return A
        end
        
        return A_jaco
    end

    # tailored for CCs 
    function create_G_func(M, N, q)

        function G_func(x)

            length_x = size(N, 2)
            x_vars = x[1:length_x]
            y_vars = x[length_x+1:end]

            G_val = N*x_vars + M*y_vars + q

            return vec(G_val)
        end

        return G_func
    end

    function create_G_jaco(M, N, q)

        function G_jaco(x)

            J_G = hcat(N, M)

            return J_G
        end

        return G_jaco
    end

    function create_H_func(M, N, q)

        function H_func(x)

            length_x = size(N, 2)
            y_vars = x[length_x+1:end]

            H_val = y_vars

            return H_val
        end

        return H_func
    end

    function create_H_jaco(M, N, q)

        function H_jaco(x)

            length_x = size(N, 2)
            length_y = size(M, 2)

            J_H = hcat(zeros(length_y, length_x), I(length_y))

            return J_H
        end

        return H_jaco
    end

end # module