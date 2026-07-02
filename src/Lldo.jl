module Lldo

    include("utilities/box_set.jl")       
    include("utilities/disjunctive_set.jl")  
    include("utilities/double_env.jl") 
    include("utilities/func_struct.jl")    
    include("utilities/nlp_helper.jl") 

    include("algorithms/smooth_alm.jl")
    include("algorithms/smooth_homotopy.jl")
    include("algorithms/class_alm.jl")
    include("algorithms/nlp.jl")

    using .BoxSetUtils: BoxSet
    using .FunctionStruct: ScalarFunction, VectorFunction

    using .SmoothALM: SmoothALMOptions, smooth_alm
    using .SmoothHomotopy: SmoothHomotopyOptions, smooth_homotopy
    using .ClassALM: ClassALMOptions, class_alm
    using .NLPSolvers: direct_algencan, ScholtesOptions, scholtes

    export BoxSet
    export ScalarFunction, VectorFunction 

    export SmoothALMOptions, smooth_alm
    export SmoothHomotopyOptions, smooth_homotopy
    export ClassALMOptions, class_alm
    export direct_algencan, ScholtesOptions, scholtes

end