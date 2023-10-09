####
#   "abstraction.jl"
#   Implementation of adaptive refinement method
####


"""
    refine(sys::A, N::Int, ε::Float64; verbose::Bool = false)

Implementation of [BRAJ23, Algorithm 2]
[BRAJ23] Adrien Banse, Licio Romao, Alessandro Abate, Raphaël M. Jungers, Data-driven abstractions via adaptive refinments and a Kantorovich metric
"""
function refine(
    sys::A,
    N::Int,
    ε::Float64;
    verbose::Bool = false,
    filename::Union{String,Nothing} = nothing,
) where {A<:AbstractDynamicalSystem}
    n = ceil(Int, log2(1 / ε))

    W_current = [[l] for l in outputs(sys)]
    abs_current = (oracle(sys))(W_current)

    if verbose
        Ni = N
        println("(k = $(Ni - N)) Current abstraction:")
        println(abs_current)
    end

    p_current = compute_probabilities(abs_current, n)

    while !issetequal(Set(collect(values(P(abs_current)))), Set([0.0, 1.0]))
        if N == 0
            if !isnothing(filename)
                save_abs(abs_current, filename)
            end
            return abs_current
        end

        abs_next = nothing
        W_next = nothing
        d_next = -Inf
        p_next = nothing

        for leaf in W_current

            println(leaf)
            if leaf[end] == 1
                continue
            end

            W_test = copy(W_current)
            W_test = remove!(W_test, leaf)
            push!(W_test, [append!(copy(leaf), o) for o in outputs(sys)]...)
            abs_test = (oracle(sys))(W_test)

            p_test = compute_probabilities(abs_test, n)
            d_test = kantorovich(abs_current, abs_test, n; p_1 = p_current, p_2 = p_test)

            if d_test > d_next
                d_next = d_test
                abs_next = abs_test
                W_next = W_test
                p_next = p_test
            end
        end

        W_current = W_next
        abs_current = abs_next
        p_current = p_next
        N -= 1

        if verbose
            println("(k = $(Ni - N)) Current abstraction (chosen with d = $d_next):")
            println(abs_current)
        end
    end
end
