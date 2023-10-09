# using JuMP
# using MosekTools

# """
#     white_box_LQR(Σ, B, Q, R)

# Design a white-box LQR controller for ASLS x(t+1) = Ax(t) + Bu(t)
# with A in Σ, with ellipsoidal costs Q and R
# Note: it is sufficient for CSLSs as well
# """
# function white_box_LQR(Σ, B, Q, R)
#     dim, dim_in = size(B)
#     lower_triangular(P) = [P[i, j] for i = 1:size(P)[1] for j = 1:i]
#     solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
#     model = Model(solver)
#     @variable(model, S[1:dim, 1:dim] in PSDCone())
#     @variable(model, Y[1:dim_in, 1:dim])
#     @variable(model, t)
#     @constraint(model, S >= 0, PSDCone())
#     @constraint(model, [t; 1; lower_triangular(S)] in MOI.LogDetConeTriangle(dim))
#     @objective(model, Max, t)
#     for A in Σ
#         @constraint(
#             model,
#             [
#                 S S*A'+(B * Y)' S Y'
#                 A*S+B*Y S zeros(dim, dim) zeros(dim, dim_in)
#                 S zeros(dim, dim) inv(Q) zeros(dim, dim_in)
#                 Y zeros(dim_in, dim) zeros(dim_in, dim) inv(R)
#             ] >= 0,
#             PSDCone()
#         )
#     end
#     JuMP.optimize!(model)
#     if termination_status(model) == MOI.OPTIMAL
#         P = inv(value.(S))
#         K = value.(Y) * P
#         return K, P
#     else
#         println("The LQR problem is infeasible!")
#         return zeros(dim_in, dim), zeros(dim, dim)
#     end
# end

# We first include the implemented algorithms
include(joinpath(@__DIR__, "../src/KantorovichAbstraction.jl"))
KANT = KantorovichAbstraction

using Distributions
using Random
using LinearAlgebra

JSR = 0.7
A1 = JSR * [-0.282769 -1.19872; 1.01845 0.780967]
# A2 = JSR * [0.200933 2.2132; -0.318989 1.46324]
Σ = [A1, A2]
F(x) = A1 * x
H(x; ε = 0.1) = norm(x) <= ε
outputs = [0, 1]

function query_memory_system(init::Vector{Int}; n_samples = 1e4)
    # sample based query, n_samples samples
    global l = length(init)
    global c = 0
    for _ = 1:n_samples
        local out = []
        local x = [
            rand(Uniform(-1, 1))
            rand(Uniform(-1, 1))
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

system = KANT.DynamicalSystem(F, H, 2, [0.0, 0.0, 0.0], outputs, 0, oracle)

COMPUTE = false
if COMPUTE
    abst = KANT.refine(system, 10, 1e-5; verbose = true)
end

ANALYZE = true
if ANALYZE
    partitioning_5_iterations =
        [[0, 1], [0, 0, 1], [1, 1], [0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0], [1, 0, 1]]
    # abs = (KANT.oracle(system))(partitioning_5_iterations)

    # visualize the state-space
    function identify_partitions(partitioning::Vector{Vector{Int}}; n_samples = 1e5)
        res = Dict{Vector{Int},Vector{Vector{Float64}}}()

        for w in partitioning
            global l = length(w)

            for _ = 1:n_samples
                local x_init = [
                    rand(Uniform(-1, 1))
                    rand(Uniform(-1, 1))
                ]
                local x = x_init
                local out = []

                append!(out, H(x))
                for _ = 1:(l-1)
                    x = F(x)
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

    using LazySets

    # ch_array = []
    # for (k, v) in dict
    #     global ch_array
    #     q1 = Vector{Vector{Float64}}()
    #     q2 = Vector{Vector{Float64}}()
    #     q3 = Vector{Vector{Float64}}()
    #     q4 = Vector{Vector{Float64}}()
    #     for p ∈ v
    #         if p[1] >= 0 && p[2] >= 0
    #             push!(q1, p)
    #         elseif p[1] >= 0 && p[2] <= 0
    #             push!(q2, p)
    #         elseif p[1] <= 0 && p[2] >= 0
    #             push!(q3, p)
    #         else
    #             push!(q4, p)
    #         end 
    #     end

    #     println(isempty(q1))
    #     println(isempty(q2))
    #     println(isempty(q3))
    #     println(isempty(q4))

    #     for q = [q1, q2, q3, q4]
    #         if !isempty(q)
    #             push!(ch_array, (k, convex_hull(q)))
    #         end
    #     end
    # end

    ch_array = [(k, convex_hull(v)) for (k, v) in dict]
    # I should do clustering

    isless_ch(a, b) = area(VPolygon(a[2])) > area(VPolygon(b[2]))
    sort!(ch_array; lt = isless_ch)

    p = plot()
    for (k, ch) in ch_array
        lab = "$k"
        col = random_color()
        plot!(p, VPolygon(ch), alpha = 1, color = col)
    end

    # for (i, (k, v)) in enumerate(dict)
    #     lab = "$k"
    #     col = random_color()

    #     ch = convex_hull(v)
    #     plot!(p, VPolygon(ch), alpha = 1, color = col)

    #     # scatter!(
    #     #     p,
    #     #     [point[1] for point in v],
    #     #     [point[2] for point in v],
    #     #     label = lab,
    #     #     color = col,
    #     #     markersize = 0.7,
    #     #     markerstrokewidth = 0,
    #     # )
    # end
    display(p)
end
