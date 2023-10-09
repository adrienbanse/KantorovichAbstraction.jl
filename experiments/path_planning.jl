# We first include the implemented algorithms
include(joinpath(@__DIR__, "../src/KantorovichAbstraction.jl"))
KANT = KantorovichAbstraction

using Distributions
using Random

function F_cont(x, u)
    α = atan(tan(u[2]) / 2)
    return [u[1] * cos(α + x[3]) / cos(α), u[1] * sin(α + x[3]) / cos(α), u[1] * tan(u[2])]
end
# x_dot = F(x, u)
# (x' - x) / h = F(x, u)
# x' = x + h * F(x, u)
outputs = [0, 1]
is_in_rectangle_2d(x, x1_lb, x1_ub, x2_lb, x2_ub) =
    x[1] >= x1_lb && x[1] <= x1_ub && x[2] >= x2_lb && x[2] <= x2_ub
function H(x)
    X1_lb = [1.0, 2.2, 2.2]
    X1_ub = [1.2, 2.4, 2.4]
    X2_lb = [0.0, 0.0, 6.0]
    X2_ub = [9.0, 5.0, 10.0]
    for bounds in zip(X1_lb, X1_ub, X2_lb, X2_ub)
        if is_in_rectangle_2d(x[1:2], bounds...)
            return 1
        end
    end
    return 0
end

function F(x, u; h = 0.1)
    return H(x) == 1 ? x : x .+ h * F_cont(x, u)
end


function query_memory_system(init::Vector{Int}; n_samples = 1e4)
    # sample based query, n_samples samples
    global l = length(init)
    global c = 0
    for _ = 1:n_samples
        local out = []
        local x = [rand(Uniform(0, 4)), rand(Uniform(0, 10)), rand(Uniform(0, 2 * π))]
        local u = [
            rand(Uniform(-1, 1))
            rand(Uniform(-1, 1))
        ]

        append!(out, H(x))
        for _ = 1:(l-1)
            x = F(x, u)
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

system = KANT.DynamicalSystem(F, H, 3, [0.0, 0.0, 0.0], outputs, 0, oracle)

COMPUTE = false
if COMPUTE
    abs = KANT.refine(system, 7, 1e-5; verbose = true)
end

ANALYZE = true
if ANALYZE
    partitioning_5_iterations = [
        [1],
        [0, 1],
        [0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ]
    # abs = (KANT.oracle(system))(partitioning_5_iterations)

    # visualize the state-space
    function identify_partitions(partitioning::Vector{Vector{Int}}; n_samples = 1e5)
        res = Dict{Vector{Int},Vector{Vector{Float64}}}()

        for w in partitioning
            global l = length(w)

            for _ = 1:n_samples
                local x_init =
                    [rand(Uniform(0, 4)), rand(Uniform(0, 10)), rand(Uniform(0, 2 * π))]
                local u_init = [
                    rand(Uniform(-1, 1))
                    rand(Uniform(-1, 1))
                ]
                local x = x_init
                local u = u_init
                local out = []

                append!(out, H(x))
                for _ = 1:(l-1)
                    x = F(x, u)
                    append!(out, H(x))
                end

                if out == w
                    if haskey(res, w)
                        append!(res[w], [x_init])
                    else
                        res[w] = [x_init]
                    end
                end
            end
        end
        return res
    end

    dict = identify_partitions(partitioning_5_iterations)

    using Plots
    random_color() = RGB(rand(), rand(), rand())
    p = scatter()
    colors = ["blue", "red", "green", "orange", "magenta", "cyan", "pink"]
    for (i, (k, v)) in enumerate(dict)
        lab = "$k"
        col = random_color()
        scatter!(
            p,
            [point[1] for point in v],
            [point[2] for point in v],
            label = lab,
            color = col,
            markersize = 1,
            markerstrokewidth = 0,
        )
    end
    display(p)
end
