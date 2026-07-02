module mod_mpvlcc

using LinearAlgebra

export create_f_func,
       create_f_grad,
       create_F_func,
       create_F_jaco,
       create_A_func,
       create_A_jaco,
       create_f_CC_func,
       create_f_CC_grad,
       create_G_func,
       create_G_jaco,
       create_H_func,
       create_H_jaco,
       create_A_CC_func,
       create_A_CC_jaco
    
    function create_f_func(n, l)

        function f_func(x)
            return 0.5 * norm(x)^2 
        end

        return f_func
    end
        
    function create_f_grad(n, l)
        
        function f_grad(x)
            return x 
        end
        
        return f_grad
    end

    function create_F_func(n, l, i)

        function F_func(x)
            z = x[1 : n]
            w = x[n + 1 : end]

            g = zeros(l + 1)

            g[1] = z[i]

            for j in 1 : l
                g[1 + j] = w[(j - 1) * n + i]
            end
            
            return g 
        end
        
        return F_func
    end
        
    function create_F_jaco(n, l, i)
        
        function F_jaco(x)
            J = zeros(l + 1, n + n * l)  
            
            J[1, i] = 1.0  
            
            for j in 1 : l
                J[1 + j, j * n + i] = 1.0
            end
            
            return J
        end
        
        return F_jaco
    end
        
    # (M -I)x + q = 0
    function create_A_func(n, l, M, q)

        length_w = n * l
        A = [M -Matrix(1.0I, length_w, length_w)]

        function A_func(x)
            return vec(A*x + q)
        end

        return A_func
    end

    function create_A_jaco(n, l, M, q)

        length_w = n * l
        A = [M -Matrix(1.0I, length_w, length_w)]

        function A_jaco(x)
            return A
        end
        
        return A_jaco
    end

    # tailored for CCs (by adding extra variables)
    function create_f_CC_func(n, l)

        function f_func(x)
            x_true = x[1 : n + n *l]
            return 0.5 * norm(x_true)^2 
        end

        return f_func
    end
        
    function create_f_CC_grad(n, l)
        
        function f_grad(x)
            x_true = x[1 : n + n * l]
            g = zeros(n + n * l + n * l)
            g[1 : n + n * l] = x_true
            return g
        end
        
        return f_grad
    end

    function create_G_func(n ,l)

        function G_func(x)

            G_val = x[n + n * l + 1 : end]

            return vec(G_val)
        end

        return G_func
    end

    function create_G_jaco(n ,l)

        function G_jaco(x)

            N1 = zeros(n * l, n + n * l)
            N2 = Matrix(1.0I, n * l, n * l)

            J_G = hcat(N1, N2)

            return J_G
        end

        return G_jaco
    end

    function create_H_func(n ,l)

        function H_func(x)

            H_val = zeros(n * l)

            for i in 1 : l - 1

                wi = x[i * n + 1 : i * n + n]

                for k in i + 1 : l
                    sk = x[n + n * l + (k - 1) * n + 1 : n + n * l + (k - 1)* n + n]
                    wi = wi .- sk
                end

                H_val[(i - 1) * n + 1 : (i - 1) * n + n] = wi
            end

            wl = x[l * n + 1 : l * n + n]
            H_val[(l - 1) * n + 1 : (l - 1) * n + n] = wl

            return H_val
        end

        return H_func
    end

    function create_H_jaco(n ,l)

        function H_jaco(x)

            J_H = zeros(n * l, n + n * l + n * l)

            for i in 1 : l - 1

                J_H_temp = zeros(n, n + n * l + n * l)

                J_H_temp[:, i * n + 1 : i * n + n] = Matrix(1.0I, n, n)

                for k in i + 1 : l
                    J_H_temp[:, n + n * l + (k - 1) * n + 1 : n + n * l + (k - 1)* n + n] = -Matrix(1.0I, n, n)
                end

                J_H[(i - 1) * n + 1 : (i - 1) * n + n, :] = J_H_temp

            end

            J_H_temp = zeros(n, n + n * l + n * l)
            J_H_temp[:, l * n + 1 : l * n + n] = Matrix(1.0I, n, n)
            J_H[(l - 1) * n + 1 : (l - 1) * n + n, :] = J_H_temp

            return J_H
        end

        return H_jaco
    end

    # (M -I 0)x + q = 0
    # [I 0 -(I,…,I)]x = 0
    function create_A_CC_func(n, l, M, q)

        length_w = n * l
        A1 = [M -Matrix(1.0I, length_w, length_w) zeros(n * l, n * l)]
        A2 = [Matrix(1.0I, n, n) zeros(n, n * l)]

        for i in 1 : l
            A2 = [A2 -Matrix(1.0I, n, n)]
        end

        function A_func(x)
            A = [A1; A2]
            b = [q; zeros(n)]
            return vec(A*x + b)
        end

        return A_func
    end

    function create_A_CC_jaco(n, l, M, q)

        length_w = n * l
        A1 = [M -Matrix(1.0I, length_w, length_w) zeros(n * l, n * l)]
        A2 = [Matrix(1.0I, n, n) zeros(n, n * l)]

        for i in 1 : l
            A2 = [A2 -Matrix(1.0I, n, n)]
        end

        function A_jaco(x)
            A = [A1; A2]
            return A
        end
        
        return A_jaco
    end

end # module