include(joinpath(@__DIR__, "../src/KantorovichAbstraction.jl"))
KANT = KantorovichAbstraction

function compute_safety_probability(abs::KANT.MarkovChain, n_steps::Int, safe::Int, unsafe::Int)
    function rec(steps_left::Int, current::KANT.PartitionState, prod::Float64)
        if abs.L[current] == unsafe || prod == 0.
            return 0.
        end
        if steps_left == 0
            return abs.μ[current] * prod
        end
        sum_rec = 0.0
        for state ∈ KANT.S(abs)
            if haskey(abs.P, (state, current))
                r = rec(steps_left - 1, state, prod * abs.P[state, current])
                sum_rec += r
            end
        end
        return sum_rec
    end

    safe_sates = [s for s ∈ KANT.S(abs) if KANT.L(abs)[s] == safe]
    sum_tot = 0.0
    for state ∈ safe_sates
        sum_tot += rec(n_steps, state, 1.)
    end
    return sum_tot
end

FILENAME_REFINE = "data_saved/lorentz_3_0.001.jld2"
abs_refine = KANT.load_abs(FILENAME_REFINE)

FILENAME_UNIFORM = "data_saved/lorentz_uniform_16.jld2"
abs_uniform = KANT.load_abs(FILENAME_UNIFORM)

# compute the measure of finishing at 0 without going to 1 in 10 steps

safety_refine = []
safety_uniform = []
n_steps_range = collect(1:10)

for n_steps ∈ n_steps_range
    @show n_steps
    s_r = compute_safety_probability(abs_refine, n_steps, 0, 1)
    println(s_r)
    push!(safety_refine, s_r)
    s_u = compute_safety_probability(abs_uniform, n_steps, 0, 1)
    println(s_u)
    push!(safety_uniform, s_u)
    println()
end

using Plots
p = scatter(n_steps_range, safety_refine; label = "Smart")
xlabel!(p, "Number of steps")
ylabel!(p, "Probability of safety")
scatter!(p, n_steps_range, safety_uniform; label = "Uniform")
display(p)