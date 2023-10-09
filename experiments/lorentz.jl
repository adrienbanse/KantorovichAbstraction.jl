# We first include the implemented algorithms
include(joinpath(@__DIR__, "../src/KantorovichAbstraction.jl"))
KANT = KantorovichAbstraction

using Distributions
using Random

function F_cont(x)
    Q = -1.0
    M = 9.1093837
    E1 = -100.0
    E2 = 50.0
    B = 10.0
    A = [
        0.0 Q/M*B 0.0 0.0
        -Q/M*B 0.0 0.0 0.0
        1.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
    ]
    b = [
        Q / M * E1
        Q / M * E2
        0.0
        0.0
    ]
    return A * x + b
end
# x_dot = F(x)
# (x' - x) / h = F(x)
# x' = x + h * F(x)

outputs = [0, 1, 2, 3]
function H(x)
    pos = x[3:4]
    if pos[1] >= 3.0
        return 0
    end
    if pos[1] <= 1.5 && pos[1] >= 0.5 && pos[2] <= 0.5 && pos[2] >= -0.5
        return 1
    end
    if pos[2] <= 0.0
        return 2
    end
    return 3
end

function F(x; h = 0.1)
    return x .+ h * F_cont(x)
end

# function query_memory_system(init::Vector{Int}; n_samples = 1e5)
#     # sample based query, n_samples samples
#     global l = length(init)
#     global c = 0
#     for _ = 1:n_samples
#         local out = []
#         local x = [
#             rand(Uniform(-1., 1.)), 
#             rand(Uniform(-1., 4.)), 
#             rand(Uniform(-1., 1.)), 
#             rand(Uniform(-1., 1.))
#         ]
#         append!(out, H(x))

#         for _ = 1:(l - 1)
#             x = F(x)
#             append!(out, H(x))
#         end
#         c = (out == init) ? c + 1 : c
#     end
#     return c / n_samples
# end

# function query_memory_system(from::Vector{Int}, to::Vector{Int}; n_samples = 10e5)
#     from_future = from[2:length(from)]
#     shortest = min(length(from_future), length(to))
#     if from_future[1:shortest] != to[1:shortest] # no match
#         return 0.0
#     end
#     if length(from_future) >= length(to) # ex: 100 -> 00 (w.p. 1)
#         return 1.0
#     end
#     inter = vcat([from[1]], to)

#     global l_inter = length(inter)
#     global c_inter = 0

#     global l_from = length(from)
#     global c_from = 0

#     for _ = 1:n_samples
#         local out = []
#         local x = [
#             rand(Uniform(-1.0, 1.0)),
#             rand(Uniform(-1.0, 1.0)),
#             rand(Uniform(-1.0, 4.0)),
#             rand(Uniform(-1.0, 1.0)),
#         ]
#         append!(out, H(x))

#         for _ = 1:(l_inter-1)
#             x = F(x)
#             append!(out, H(x))
#         end

#         c_from = out[1:l_from] == from ? c_from + 1 : c_from
#         c_inter = out == inter ? c_inter + 1 : c_inter
#     end

#     return c_from / n_samples, c_from == 0.0 ? c_from : c_inter / c_from
# end

function oracle(W::Vector{Vector{Int}}; n_samples = 10e4)
    S = [KANT.PartitionState(w) for w in W]
    μ = Dict{KANT.PartitionState,Float64}()
    P = Dict{Tuple{KANT.PartitionState,KANT.PartitionState},Float64}()
    L = Dict{KANT.PartitionState,Int}()

    # compute largest length to sample --> compute all inter
    inters = Dict{Tuple{KANT.PartitionState,KANT.PartitionState},Vector{Int}}()
    global l_sample = 0
    for s_from in S
        from = KANT.id(s_from)
        for s_to in S
            to = KANT.id(s_to)
            from_future = from[2:length(from)]
            shortest = min(length(from_future), length(to))
            if from_future[1:shortest] != to[1:shortest] # no match
                continue
            end
            inters[s_from, s_to] = vcat([from[1]], to)
            l_sample = max(l_sample, length(inters[s_from, s_to]))
        end
    end

    # sample
    global output_list = []
    for _ = 1:n_samples
        local out = []
        local x = [
            rand(Uniform(-1.0, 1.0)),
            rand(Uniform(-1.0, 1.0)),
            rand(Uniform(-1.0, 4.0)),
            rand(Uniform(-1.0, 1.0)),
        ]
        append!(out, H(x))
        for _ = 1:(l_sample-1)
            x = F(x)
            append!(out, H(x))
        end
        push!(output_list, out)
    end

    # count 
    for ((s_from, s_to), inter) ∈ inters
        local l_from = length(KANT.id(s_from))
        local c_from = 0
        local l_inter = length(inter)
        local c_inter = 0
        for out ∈ output_list
            c_from = out[1:l_from] == KANT.id(s_from) ? c_from + 1 : c_from
            c_inter = out[1:l_inter] == inter ? c_inter + 1 : c_inter
        end
        if !haskey(μ, s_from)
            μ[s_from] = c_from / n_samples
            L[s_from] = KANT.id(s_from)[1]
        end
        P[s_from, s_to] = c_from == 0.0 ? c_from : c_inter / c_from
    end

    labels = outputs
    return KANT.MarkovChain{KANT.PartitionState}(S, P, μ, labels, L)
end

system = KANT.DynamicalSystem(F, H, 4, [0.0, 0.0, 0.0, 0.0], outputs, 2, oracle)
# abs_ref = system.oracle([[o] for o ∈ outputs])

n_steps = 3
tol = 1e-3
partition = [
    [0, 0], [0, 1], [0, 2], [0, 3],
    [1, 0], [1, 1], [1, 2], [1, 3], 
    [2, 0], [2, 1], [2, 2], [2, 3], 
    [3, 0], [3, 1], [3, 2], [3, 3], 
]

# abs = system.oracle(partition)
KANT.save_abs(abs, "data_saved/lorentz_uniform_16")
# abs = KANT.refine(system, n_steps, tol; verbose = true)#, filename = "data_saved/lorentz_$(n_steps)_$(tol)")
# abs = KANT.load_abs("data_saved/lorentz_$(n_steps)_$(tol)")

# Now construct the uniform partitioning












ANALYZE = false
if ANALYZE
    # visualize the state-space
    function identify_partitions(partitioning::Vector{Vector{Int}}; n_samples = 1e5)
        res = Dict{Vector{Int},Vector{Vector{Float64}}}()

        for w in partitioning
            global l = length(w)

            for _ = 1:n_samples
                local x_init = [
                    rand(Uniform(-1.0, 1.0)),
                    rand(Uniform(-1.0, 1.0)),
                    rand(Uniform(-1.0, 4.0)),
                    rand(Uniform(-1.0, 1.0)),
                ]
                local x = x_init
                local out = []

                append!(out, H(x))
                for _ = 1:(l - 1)
                    x = F(x)
                    append!(out, H(x))
                end

                if out == w
                    if haskey(res, w)
                        push!(res[w], x_init)
                    else
                        res[w] = [x_init]
                    end
                end
            end
        end
        return res
    end

    dict = identify_partitions([[0], [1], [3, 0], [3, 1], [3, 2], [3, 3, 0], [3, 3, 1], [3, 3, 2], [3, 3, 3], [2, 0], [2, 1], [2, 2], [2, 3]])

    using Plots
    random_color() = RGB(rand(), rand(), rand())
    p = scatter()
    for (i, (k, v)) in enumerate(dict)
        lab = "$k"
        col = random_color()
        scatter!(
            p,
            [point[3] for point in v],
            [point[4] for point in v],
            label = lab,
            color = col,
            markersize = 1,
            markerstrokewidth = 0,
            legend = false
        )
    end
    display(p)
end
