####
#   "utils.jl"
#   Implementation of util functions
####

function remove!(a, item)
    deleteat!(a, findall(x -> x == item, a))
end

using JLD2
function save_abs(abs::MarkovChain, filename::String)
    filename = endswith(filename, ".jld2") ? filename : "$filename.jld2"
    save_object(filename, abs)
end
function load_abs(filename::String)
    filename = endswith(filename, ".jld2") ? filename : "$filename.jld2"
    return load_object(filename)
end

function filter(dict)
    d = Dict()
    for (k, v) ∈ dict
        if v != 0.0
            d[k] = v
        end
    end
    return d
end