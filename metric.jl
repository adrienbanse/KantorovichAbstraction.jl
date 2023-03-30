function wasserstein(
    μ1::Dict{Vector{Int}, Float64}, 
    μ2::Dict{Vector{Int}, Float64},
    P1::Dict{Tuple{Vector{Int}, Vector{Int}}, Float64}, 
    P2::Dict{Tuple{Vector{Int}, Vector{Int}}, Float64}, 
    m::Int,
    horizon::Int
)
    function markov_wasserstein_rec(
        n::Int, 
        parent::Vector{Int},
        left::Float64
    )   
        if n == horizon || left == 0
            return 0
        end

        n += 1
        p1_vec = zeros(m)
        p2_vec = zeros(m)
        left_vec = zeros(m)
        for i = 0:(m - 1)
            word = vcat(parent, [i])
            p1_vec[i + 1] = compute_probability(P1, μ1, word, m)
            p2_vec[i + 1] = compute_probability(P2, μ2, word, m)

            left_vec[i + 1] = min(p1_vec[i + 1], p2_vec[i + 1])
        end
        next_sum = (2.)^(-n) * (left - sum(left_vec))

        for i = 0:(m - 1)
            if left_vec[i + 1] != 0
                next_sum += markov_wasserstein_rec(
                    n,
                    append!(copy(parent), i),
                    left_vec[i + 1]
                )
            end
        end
        return next_sum
    end

    return markov_wasserstein_rec(0, Vector{Int}(), 1.)
end

function find_knowledge(leafs::Vector{Vector{Int}}, word::Vector{Int}, m::Int)
    max_w = nothing
    w = Vector{Int}()
    l = length(word)
    for c = word[1:l]
        w_try = vcat(w, [c])
        if w_try in leafs
            max_w = copy(w_try)
        end
        w = w_try
    end
    if !isnothing(max_w)
        return [max_w]
    end

    # word can be expressed as sum of some leaves
    function collect_leaves!(res, parent)
        for i = 0:(m - 1)
            leaf_try = vcat(parent, [i])
            if leaf_try in leafs
                append!(res, [leaf_try])
            else
                collect_leaves!(res, leaf_try)
            end
        end
    end
    res = Vector{Vector{Int}}()
    collect_leaves!(res, word)
    return res
end

function compute_probability(
    P::Dict{Tuple{Vector{Int}, Vector{Int}}, Float64}, 
    μ::Dict{Vector{Int}, Float64}, 
    word::Vector{Int}, 
    m::Int
)
    leafs = collect(keys(μ))

    w1_l = find_knowledge(leafs, word, m)
    if length(w1_l) != 1
        return sum([μ[w] for w = w1_l])
    end

    w1 = w1_l[1]
    p = μ[w1]

    l = length(word)
    for i = 2:l
        w2_l = find_knowledge(leafs, word[i:l], m)
        if length(w2_l) != 1
            return p * sum([P[w1, w2] for w2 = w2_l])
        end
        w2 = w2_l[1]
        p *= P[w1, w2]
        w1 = w2
    end
    return p
end

