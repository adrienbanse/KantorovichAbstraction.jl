using Plots
using Compose

# function chain_to_lightgraph(chain)
#     g = LightGraphs.DiGraph(state_count(chain))
#     for tr in transitions(chain)
#         LightGraphs.add_edge!(g, tr.src, tr.dst)
#     end
#     return g
# end

# Inspired from https://github.com/wangnangg/MarkovChains.jl/blob/master/src/plot.jl
@recipe function f(mc::MarkovChain{S}) where {S<:AbstractState}
    
end
