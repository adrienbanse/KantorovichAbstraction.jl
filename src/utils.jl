####
#   "utils.jl"
#   Implementation of util functions
####

function remove!(a, item)
    deleteat!(a, findall(x->x==item, a))
end