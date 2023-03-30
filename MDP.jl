include(joinpath(@__DIR__, "Wasserstein.jl"))
K = Wasserstein

sys = K.memory_system()

_actions = [0, 1/4, 1/2]
_reward = function (s, a) return s == 0 ? 1. : 0. end
_discount = .95

function compute_reward_system(samples, discount, reward)
    N, L = size(samples)
    global r_gat = 0
    for n = 1:N
        local r = 0
        for l = 1:L
            r += discount^(l - 1) * reward(samples[n, l], nothing)
        end
        r_gat += r
    end
    return r_gat / N
end

function print_policy(policy::Policy, states)
    for s = states
        println("For $s: $(action(policy, s))")
    end
end

function solve_MDP_abstraction(
    __states, 
    __initial_probs, 
    __transition;
    verbose = false
)
    m = QuickMDP(
        states = __states,
        actions = _actions,
        initialstate = SparseCat(__states, __initial_probs),
        discount = _discount,
        transition = __transition,
        reward = _reward
    )
    solver = ValueIterationSolver(max_iterations=10000)
    policy = solve(solver, m)
    w = 0

    if verbose
        println("-- Result --")
        for (i, s) = enumerate(states(m))
            println("V($s) = $(value(policy, s))")
            w += __initial_probs[i] * value(policy, s)
        end
    end

    return policy
end

_states = [0, 1]
_initial_probs = [1/2, 1/2]
function _transition(s, a)
    res = Dict(
        0 => Deterministic(0), 
        1 => SparseCat([1, 0], [7/8, 1/8])
    )
    return res[s]
end
p = solve_MDP_abstraction(_states, _initial_probs, _transition)
print_policy(p, _states)
cont_sys = K.memory_system_mdp_controlled(p, _states)
samples = K.generate_trajectories(cont_sys, 1000, 5000)
println(
    "Expected reward = $(compute_reward_system(samples, _discount, _reward))\n"
)

_states = [0, 10, 11]
_initial_probs = [1/2, 1/16, 7/16]
function _transition(s, a)
    if a == 0
        res = Dict(
            0 => Deterministic(0), 
            10 => Deterministic(0),
            11 => SparseCat([11, 10], [6/7, 1/7])
        )
        return res[s]
    end
    res = Dict(
        0 => Deterministic(0),
        10 => Deterministic(11),
        11 => SparseCat([11, 10, 0], [5/7, 1/7, 1/7])
    )
    return res[s]
end
p = solve_MDP_abstraction(_states, _initial_probs, _transition)
print_policy(p, _states)
cont_sys = K.memory_system_mdp_controlled(p, _states)
samples = K.generate_trajectories(cont_sys, 1000, 5000)
println(
    "Expected reward = $(compute_reward_system(samples, _discount, _reward))\n"
)

_states = [0, 10, 110, 111]
_initial_probs = [1/2, 1/16, 1/16, 3/8]
function _transition(s, a)
    if a == 0
        res = Dict(
            0 => Deterministic(0),
            10 => Deterministic(0),
            110 => Deterministic(10),
            111 => SparseCat([111, 110], [2/3, 1/3])
        )
        return res[s]
    end
    if a == 1/4
        res = Dict(
            0 => Deterministic(0),
            10 => Deterministic(111),
            110 => Deterministic(111),
            111 => SparseCat([111, 110, 10, 0], [1/3, 1/3, 1/6, 1/6])
        )
        return res[s]
    end
    if a == 1/2
        res = Dict(
            0 => Deterministic(0),
            10 => Deterministic(110),
            110 => Deterministic(110),
            111 => SparseCat([111, 10, 0], [2/3, 1/6, 1/6])
        )
        return res[s]
    end
end
p = solve_MDP_abstraction(_states, _initial_probs, _transition)
print_policy(p, _states)
cont_sys = K.memory_system_mdp_controlled(p, _states)
samples = K.generate_trajectories(cont_sys, 1000, 5000)
println(
    "Expected reward = $(compute_reward_system(samples, _discount, _reward))\n"
)

_states = [0, 10, 110, 1110, 1111]
_initial_probs = [1/2, 1/16, 1/16, 1/8, 2/8]
function _transition(s, a)
    if a == 0
        res = Dict(
            0 => Deterministic(0),
            10 => Deterministic(0),
            110 => Deterministic(10),
            1110 => Deterministic(110),
            1111 => Deterministic(1111),
        )
        return res[s]
    end
    if a == 1/4
        res = Dict(
            0 => Deterministic(0),
            10 => Deterministic(1111),
            110 => Deterministic(1111),
            1110 => Deterministic(1111),
            1111 => SparseCat([110, 10, 0], [1/2, 1/4, 1/4])
        )
        return res[s]
    end
    if a == 1/2
        res = Dict(
            0 => Deterministic(0),
            10 => Deterministic(110),
            110 => Deterministic(110),
            1110 => SparseCat([10, 0], [1/2, 1/2]),
            1111 => Deterministic(1111)
        )
        return res[s]
    end
end
p = solve_MDP_abstraction(_states, _initial_probs, _transition)
print_policy(p, _states)
cont_sys = K.memory_system_mdp_controlled(p, _states)
samples = K.generate_trajectories(cont_sys, 1000, 5000)
println(
    "Expected reward = $(compute_reward_system(samples, _discount, _reward))\n"
)