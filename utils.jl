function make_stochastic!(P::Dict{Tuple{Vector{Int}, Vector{Int}}, F}) where {F <: AbstractFloat}
    row_sum = Dict{Vector{Int}, Float64}()
    for ((w1, _), v) in P
        if haskey(row_sum, w1)
            row_sum[w1] += v
        else
            row_sum[w1] = v
        end
    end
    for ((w1, w2), _) in P
        if row_sum[w1] != 0
            P[(w1, w2)] /= row_sum[w1]
        end
    end
end

function make_stochastic!(μ::Dict{Int, F}) where {F <: AbstractFloat}
    s = sum(values(μ))
    for (w, _) in μ
        μ[w] /= s
    end
end

function max_knowledge(parent::Vector{Int}, target::Int, P::Dict{Tuple{Vector{Int}, Vector{Int}}, F}) where {F <: AbstractFloat}
    longest = Vector{Int}()
    for ((w1, w2), _) = P
        if w2[1] == target
            if endswith(join(string.(parent)), join(string.(w1))) && length(w1) > length(longest)
                longest = w1
            end
        end
    end
    return longest
end

function remove!(a, item)
    deleteat!(a, findall(x->x==item, a))
end

function print_results(
    μ::Dict{Vector{Int}, Float64}, 
    P::Dict{Tuple{Vector{Int}, Vector{Int}}, Float64}, 
    d::Float64
)
    println("-------------------------")
    println("Partitions are ")
    for (k, v) in μ 
        println("$k (with initial prob. $v)")
    end
    println("(with d = $d)")
    println("And transition probabilities are")
    for ((k1, k2), v) in P
        if v != 0
            println("$k1 -> $k2 with probability $v")
        end
    end
    println("-------------------------\n")
end

function print_results(
    μ::Dict{Vector{Int}, Float64}, 
    P::Dict{Tuple{Vector{Int}, Vector{Int}}, Float64}
)
    println("-------------------------")
    println("Partitions are ")
    for (k, v) in μ 
        println("$k (with initial prob. $v)")
    end
    println("And transition probabilities are")
    for ((k1, k2), v) in P
        if v != 0
            println("$k1 -> $k2 with probability $v")
        end
    end
    println("-------------------------\n")
end