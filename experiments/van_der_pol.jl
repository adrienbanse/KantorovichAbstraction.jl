# We first include the implemented algorithms
include(joinpath(@__DIR__, "../src/KantorovichAbstraction.jl"))
KANT = KantorovichAbstraction

using Distributions
using Random

μ = 2.0
h = 0.1

F(x::Vector{Float64}) =
    [h * x[2] + x[1], x[2] + h * μ * (1 - x[1] * x[1]) * x[2] - h * x[1]]

# Output for an approximation of the absorbing set
outputs = [0, 1]
H(x::Vector{Float64}) = x[1] <= 0.5 && x[2] <= 0.5 ? 1 : 0

dim = 2
x_0 = [0.0, 0.0]

function query_memory_system(init::Vector{Int}; n_samples = 1e4)
    # sample based query, n_samples samples
    global l = length(init)
    global c = 0
    for _ = 1:n_samples
        local out = []
        local x = [
            rand(Uniform(0, 1)),
            rand(Uniform(0, 1)),
            rand(Uniform(0, 1)),
            rand(Uniform(0, 1)),
        ]
        append!(out, H(x))
        for _ = 1:(l-1)
            x = F(x)
            append!(out, H(x))
        end
        c = (out == init) ? c + 1 : c
    end
    return c / n_samples
end

function query_memory_system(from::Vector{Int}, to::Vector{Int})::Float64
    p_from = query_memory_system(from)
    p_to = query_memory_system(to)
    if p_from == 0.0 || p_to == 0.0
        return 0.0
    end
    from_future = from[2:length(from)]
    shortest = min(length(from_future), length(to))
    if from_future[1:shortest] != to[1:shortest] # no match
        return 0.0
    end
    if length(from_future) >= length(to) # ex: 100 -> 00 (w.p. 1)
        return 1.0
    end
    inter = vcat([from[1]], to)
    return query_memory_system(inter) / p_from
end

function oracle(W::Vector{Vector{Int}})
    S = [KANT.PartitionState(w) for w in W]
    μ = Dict{KANT.PartitionState,Float64}()
    P = Dict{Tuple{KANT.PartitionState,KANT.PartitionState},Float64}()
    L = Dict{KANT.PartitionState,Int}()
    for s_from in S
        μ[s_from] = query_memory_system(KANT.id(s_from))
        L[s_from] = KANT.id(s_from)[1]
        for s_to in S
            P[s_from, s_to] = query_memory_system(KANT.id(s_from), KANT.id(s_to))
        end
    end
    labels = outputs
    return KANT.MarkovChain{KANT.PartitionState}(S, P, μ, labels, L)
end

system = KANT.DynamicalSystem(F, H, dim, x_0, outputs, H(x_0), oracle)
# abs = (KANT.oracle(system))([[0], [1]])

KANT.refine(system, typemax(Int), 1e-5; verbose = true)

# function compare(W1, W2)
#     abs1 = (K.oracle(system))(W1)
#     p1 = K.compute_probabilities(abs1, n)
#     abs2 = (K.oracle(system))(W2)
#     p2 = K.compute_probabilities(abs2, n)
#     k = K.kantorovich(abs1, abs2, n; p_1 = p1, p_2 = p2)
#     println("d($W1, $W2) = $k")
# end

# compare([[0], [1, 0], [1, 1]], [[0, 0], [0, 1], [1, 0], [1, 1]])
# compare([[0], [1, 0], [1, 1]], [[0], [1, 0, 0], [1, 0, 1], [1, 1]])

# for W = [
#     ["0000", "0001", "001", "01", "1"], 
#     ["000", "0010", "0011", "01", "1"],
#     ["000", "001", "010", "011", "1"],
#     ["000", "001", "01", "10", "11"],
#     ["000", "001", "010", "011", "1"],
#     ["00", "0100", "0101", "011", "1"],
#     ["00", "010", "0110", "0111", "1"],
#     ["00", "010", "011", "10", "11"],
#     ["000", "001", "010", "011", "1"],
#     ["00", "0100", "0101", "011", "1"],
#     ["00", "010", "0110", "0111", "1"],
#     ["00", "010", "011", "10", "11"],
#     ["000", "001", "010", "011", "1"],
#     ["00", "0100", "0101", "011", "1"],
#     ["00", "010", "0110", "0111", "1"],
#     ["00", "010", "011", "10", "11"],
#     ["00", "01", "100", "101", "11"],
#     ["0", "1000", "1001", "101", "11"],
#     ["0", "100", "1010", "1011", "11"],
#     ["0", "100", "101", "110", "111"],
#     ["00", "01", "10", "110", "111"],
#     ["0", "100", "101", "110", "111"],
#     ["0", "10", "1100", "1101", "111"],
#     ["0", "10", "110", "1110", "1111"],
# ]
#     global d_min
#     global abs_min
#     W = [[parse(Int64, i) for i = e] for e = W]

#     abs = (K.oracle(system))(W)
#     p = K.compute_probabilities(abs, n)

#     println(abs)
#     k = K.kantorovich(abs_system, abs, n; p_1 = p_system, p_2 = p)
#     if k < d_min
#         d_min = k 
#         abs_min = abs
#     end
#     println("d(abs_0, abs_level_2) = $k")
# end

# println("\nMin is: ")
# println(abs_min)
