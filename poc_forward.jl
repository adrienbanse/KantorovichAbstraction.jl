include(joinpath(@__DIR__, "Wasserstein.jl"))
W = Wasserstein

sys = W.memory_system()
m = 2
iter_wass = 40
println("Precision is $((2.)^(-iter_wass))\n")

# PART 1
μ = Dict{Vector{Int}, Float64}()
P = Dict{Tuple{Vector{Int}, Vector{Int}}, Float64}()

# WARNING: MODEL-BASED VERSION ONLY (with memory system)
μ[[0]] = .5
μ[[1]] = .5
for c = 0:(m - 1)
    for cp = 0:(m - 1)
        P[[c], [cp]] = W.query_memory_system([c], [cp])
    end
end

W.print_results(μ, P)

# PART 2
leafs = [[0], [1]]
max_iter = 10
for _ = 1:max_iter
    global P, μ, leafs
    next_wass = -Inf
    next_leafs = nothing
    next_P = nothing
    next_μ = nothing
    for parent_try = leafs
        local P_try, μ_try
        P_try = Dict{Tuple{Vector{Int}, Vector{Int}}, Float64}()
        μ_try = Dict{Vector{Int}, Float64}()
        leafs_try = copy(leafs)
        W.remove!(leafs_try, parent_try)
        push!(leafs_try, [append!(copy(parent_try), c) for c = 0:(m - 1)]...)
        
        # WARNING: MODEL-BASED VERSION ONLY (with memory system)
        # TO GO TO DATA-DRIVEN VERSION: change parts where W.query_memory_system appears
        for l1 = leafs_try
            μ_try[l1] = W.query_memory_system(l1)
            for l2 = leafs_try
                P_try[l1, l2] = W.query_memory_system(l1, l2)
            end
        end

        wass_try = W.wasserstein(μ, μ_try, P, P_try, m, iter_wass) 

        println(keys(μ_try))
        println(wass_try)

        if wass_try > next_wass
            next_wass = wass_try
            next_leafs = leafs_try
            next_P = copy(P_try)
            next_μ = copy(μ_try)
        end
    end
    leafs = next_leafs
    P = copy(next_P)
    μ = copy(next_μ)
    W.print_results(μ, P, next_wass)

    # stopping criterion (TODO: discuss this)
    if issubset(values(P), [0., 1.])
        break
    end
end
