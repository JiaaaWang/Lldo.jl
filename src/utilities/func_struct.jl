""" 
    function structure         
"""

module FunctionStruct

using ForwardDiff

export ScalarFunction, VectorFunction

    # represents a scalar-valued function with optional gradient
    struct ScalarFunction
        name::Symbol
        func::Function           
        grad::Union{Function, Nothing}  
    end

    # represents a vector-valued function with optional Jacobian
    struct VectorFunction
        name::Symbol
        func::Function           
        jaco::Union{Function, Nothing}  
    end
 
    function (sf::ScalarFunction)(x::AbstractVector)
        value = sf.func(x)
        
        if sf.grad !== nothing
            value_grad = sf.grad(x)  
        else
            value_grad = ForwardDiff.gradient(sf.func, x)  
        end
        
        return value, value_grad
    end

    function (vf::VectorFunction)(x::AbstractVector)
        value = vf.func(x)  
        
        if vf.jaco !== nothing
            value_jaco = vf.jaco(x)  
        else
            value_jaco = ForwardDiff.jacobian(vf.func, x)  
        end
        
        return value, value_jaco
    end

end # module