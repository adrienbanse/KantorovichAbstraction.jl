####
#   "metric.jl"
#   Definition of structure MarkovChain
#   Definition of structure BasicState (state corresponding to one integer)
#   Definition of structure PartitionState (state corresponding to a vector of integers), 
#       used for Markov chains that are abstractions where states correspond to partitions
#   Implementation of Kantorovich measure between Markov chains
####

abstract type AbstractState end
id(state::A) where A<:AbstractState = state.id 
Base.show(io::IO, state::A) where A<:AbstractState = print(io, state.id)
mutable struct BasicState <: AbstractState
    id::Int
end
mutable struct PartitionState <: AbstractState
    id::Vector{Int}
end

mutable struct MarkovChain{S<:AbstractState}
    S::Vector{S}
    P::Dict{Tuple{S, S}, Float64}
    μ::Dict{S, Float64}
    labels::Vector{Int}
    L::Dict{S, Int}
end
MarkovChain{S}() where S<:AbstractState = MarkovChain(
    Vector{S}(), 
    Dict{Tuple{S, S}, Float64}(), 
    Dict{S, Float64}(),
    Vector{Int}(), 
    Dict{S, Int}()
)
S(chain::MarkovChain) = chain.S
P(chain::MarkovChain) = chain.P
μ(chain::MarkovChain) = chain.μ
labels(chain::MarkovChain) = chain.labels
L(chain::MarkovChain) = chain.L
Base.show(io::IO, chain::MarkovChain) = print(io, "Markov Chain with states $(S(chain))")
function add_state!(
    chain::MarkovChain, 
    state::A, 
    label::Int, 
    init::Float64
) where A<:AbstractState
    append!(chain.S, [state])
    chain.μ[state] = init
    chain.L[state] = label  
    if label ∉ labels(chain)
        append!(chain.labels, [label])
    end
end
function add_transition!(chain::MarkovChain, from::A, to::A, p::Float64) where A<:AbstractState
    chain.P[from, to] = p
end

"""
    compute_words(labels::Vector{Int}, n::Int)

Given a set of "labels" and "n", compute_words computes 
all the words of length "n" defined on the alphabet "labels"
"""
function compute_words(labels::Vector{Int}, n::Int)
    function compute_words_rec!(word_list::Vector{Vector{Vector{Int}}}, k::Int)
        if k >= n return end
        word_sublist = Vector{Vector{Int}}()
        for w = word_list[k]
            for l = labels
                new_w = push!(copy(w), l)
                push!(word_sublist, new_w)
            end
        end
        push!(word_list, word_sublist)
        compute_words_rec!(res, k + 1)
    end
    res = [[[l] for l = labels]]
    compute_words_rec!(res, 1)
    return res
end

"""
    compute_probabilities(chain::MarkovChain, n::Int)

Given a Markov chain "chain" and "n", compute_probabilities computes the vectors 
p^k for k = 1, ..., "n" as defined in [BRAJ23, Equation (2)] in O(|A|^(n+1)|S|^2)
corresponding to [BRAJ23, Remark 1]
[BRAJ23] Adrien Banse, Licio Romao, Alessandro Abate, Raphaël M. Jungers, Data-driven abstractions via adaptive refinments and a Kantorovich metric
"""
function compute_probabilities(chain::MarkovChain, n::Int) 
    A = typeof(S(chain)[1])
    p_words_total = Vector{Dict{Vector{Int}, Float64}}()

    # initialization of "current" quantities
    words_current = [[l] for l = labels(chain)]
    p_words_states_current = Dict{Vector{Int}, Dict{A, Float64}}()
    p_words_current = Dict{Vector{Int}, Float64}()
    for w = words_current
        p_words_states_current[w] = Dict{A, Float64}()
        for s = S(chain)
            p_words_states_current[w][s] = (L(chain))[s] == w[length(w)] ? (μ(chain))[s] : 0.
        end
        p_words_current[w] = sum([p_words_states_current[w][s] for s = S(chain)])
    end
    push!(p_words_total, p_words_current)

    # update "current" with "next"
    for _ = 2:n
        words_next = Vector{Vector{Int}}()
        p_words_states_next = Dict{Vector{Int}, Dict{A, Float64}}()
        p_words_next = Dict{Vector{Int}, Float64}()
        for w = words_current
            for l = labels(chain)
                w_new = push!(copy(w), l)
                push!(words_next, w_new)
                p_words_states_next[w_new] = Dict{A, Float64}()
                for s_to = S(chain)
                    p_words_states_next[w_new][s_to] = 0.
                    if (L(chain))[s_to] != l 
                        continue
                    end
                    for s_from = S(chain)
                        if haskey(P(chain), (s_from, s_to))
                            p_words_states_next[w_new][s_to] += p_words_states_current[w][s_from] * P(chain)[s_from, s_to]
                        end
                    end
                end
                p_words_next[w_new] = sum([p_words_states_next[w_new][s] for s = S(chain)])
            end
        end
        push!(p_words_total, p_words_next)
        words_current = words_next
        p_words_states_current = p_words_states_next
        p_words_current = p_words_next
    end

    return p_words_total
end

"""
    kantorovich(chain_1::MarkovChain, chain_2::MarkovChain, n::Int; p_1::Vector{Dict{Vector{Int}, Float64}} = nothing, p_2::Vector{Dict{Vector{Int}, Float64}} = nothing)

Implementation of [BRAJ23, Algorithm 1]
[BRAJ23] Adrien Banse, Licio Romao, Alessandro Abate, Raphaël M. Jungers, Data-driven abstractions via adaptive refinments and a Kantorovich metric
"""
function kantorovich(
    chain_1::MarkovChain, 
    chain_2::MarkovChain, 
    n::Int; 
    p_1::Vector{Dict{Vector{Int}, Float64}} = nothing, 
    p_2::Vector{Dict{Vector{Int}, Float64}} = nothing
)
    if !issetequal(Set(labels(chain_1)), Set(labels(chain_2)))
        throw(AssertionError)
    end
    p_1 = isnothing(p_1) ? compute_probabilities(chain_1, n) : p_1
    p_2 = isnothing(p_2) ? compute_probabilities(chain_2, n) : p_2

    function kant_rec(k::Int, m::Float64, w::Vector{Int})
        r = Vector{Float64}()
        w_new_list = Vector{Vector{Int}}()
        for l = labels(chain_1)
            w_new = push!(copy(w), l)
            push!(w_new_list, w_new)
            push!(r, min(p_1[k + 1][w_new], p_2[k + 1][w_new]))
        end
        res = (2.)^(-(k + 1)) * (m - sum(r))
        if k + 1 == n
            return res
        end
        for i = 1:length(labels(chain_1))
            if r[i] != 0
                res += kant_rec(k + 1, r[i], w_new_list[i])
            end
        end
        return res
    end
    return kant_rec(0, 1., Vector{Int}())
end



