struct DynamicalSystem{Ft<:Function, Fo<:Function, Fs<:Function}
    transfer::Ft                    # transfer function
    output::Fo                      # output function
    sampling::Fs                    # sampling function
end 

function generate_trajectories(sys::DynamicalSystem, l::Int, n_samples::Int)
    x = zeros(Float64, (n_samples, l + 1, sys.dim))
    x[:, 1, :] = sys.sampling(n_samples)

    y = zeros(Int, (n_samples, l + 1))
    for sample = 1:n_samples
        y[sample, 1] = sys.output(x[sample, 1, :])
    end

    for i = 1:l
        for sample = 1:n_samples
            x[sample, i + 1, :] = (sys.transfer)(x[sample, i, :])
            y[sample, i + 1] = (sys.output)(x[sample, i + 1, :])
        end
    end
    
    y
end

# Examples
function sturmian_system()
    sturmian_dynamics(x::Vector{Float64}, θ::Irrational) = [mod(x[1] + θ, 2 * π)] 
    sturmian_output(x::Vector{Float64}, θ::Irrational) = x[1] < θ ? 0 : 1
    t = x -> sturmian_dynamics(x, Base.MathConstants.φ)
    o = x -> sturmian_output(x, Base.MathConstants.φ)
    function s(n_samples::Int)
        res = zeros(n_samples, 1)
        res[:, 1] = rand(Uniform(0, 2 * π), n_samples)
        return res
    end
    DynamicalSystem(t, o, s)
end

function switched_system()
    switched_dynamics(x::Vector{F}, Σ::Vector{Matrix{F}}) where {F <: AbstractFloat} = Σ[rand(1:size(Σ)[1])] * x
    function switched_output(x::Vector{F}, α::F)::Int where {F <: AbstractFloat}
        nm = norm(x, 2)   
        if nm > 2.
            return 8
        end

        # the following is just in case of numerical issue where 
        #   | dot([sin(α), cos(α)], x) / nm | > 1 
        # although it is not mathematically possible
        cos_val = dot([sin(α), cos(α)], x) / nm
        cos_val = abs(cos_val) > 1 ? sign(cos_val) : cos_val

        θ = acos(cos_val)

        if det([x[1] cos(α); x[2] sin(α)]) > 0
            θ = θ
        else
            θ = 2 * π - θ
        end
        part = Int(θ ÷ (π / 2))
        return norm(x, 2) > 1. ? part + 4 : part
    end

    factor = 1
    A1 = factor * [3^(1/2)/2 -1/2; 1/2 3^(1/2)/2]       # 30 degrees rotation matrix
    A2 = factor * [1.02 0; 0 1/2]                       # contraction/extension matrix
    Σ = [A1, A2]

    α = π / 5
    
    t = x -> switched_dynamics(x, Σ)
    o = x -> switched_output(x, α)

    function s(n_samples::Int)
        res = zeros(n_samples, 2)
        alpha_vec = rand(Uniform(0, 2 * π), n_samples)
        r_vec = 2 .* sqrt.(rand(Uniform(0, 1), n_samples))

        res[:, 1] = r_vec .* sin.(alpha_vec)
        res[:, 2] = r_vec .* cos.(alpha_vec)
        return res
    end

    return DynamicalSystem(t, o, s)
end

function memory_system()
    function t(x::Vector{Float64})
        if x[1] >= 1.
            return x
        elseif x[2] <= .5
            return [.5 * x[1]  + .5, x[2] + .5]
        elseif x[1] >= .5
            return [x[1] - .5, x[2]]
        elseif x[2] >= .75
            return [2. * x[1] + 1., 4. * (x[2] - .75)]
        else
            return x
            # ==> The following would be a SMALE HORSESHOE example
            #     https://en.wikipedia.org/wiki/Horseshoe_map
            # return [2. * x[1], 2. * (x[2] - .5)] 
        end
    end
    function o(x::Vector{Float64})
        if x[1] >= 1.
            return 0
        else
            return 1
        end
    end
    function s(n_samples::Int)
        res = zeros(n_samples, 2)
        res[:, 1] = rand(Uniform(0, 2), n_samples)
        res[:, 2] = rand(Uniform(0, 1), n_samples)
        return res
    end
    DynamicalSystem(t, o, s)
end

# These functions are a mess, TODO clean !!
not(c::Int) = c == 0 ? 1 : 0
function query_memory_system(init::Vector{Int})::Float64
    res_dict = Dict(
        0 => 1 / 2,
        1 => Dict(
            0 => 1 / 8,
            1 => Dict(
                0 => 1 / 7,
                1 => Dict(
                    0 => 1 / 3,
                    1 => 2 / 3
                )
            )
        )
    )
    function return_from_res(dict::Dict, init::Vector{Int}, agg::Float64)
        l = length(init)
        if l == 0 
            return agg
        end
        c = init[1]
        if typeof(dict[c]) <: Dict
            return return_from_res(dict[c], init[2:l], agg * (1 - dict[not(c)]))
        else
            return dict[c] * agg
        end
    end
    if length(init) > 1
        for (c1, c2) = zip(init[1:length(init) - 1], init[2:length(init)])
            if c1 == 0 && c2 == 1
                return 0.
            end
        end
    end
    if length(init) > 4 && init[1:4] == ones(4)
        if init == ones(length(init))
            return 1 / 4
        else
            return 0.
        end
    end
    return return_from_res(res_dict, init, 1.)
end

function query_memory_system(from::Vector{Int}, to::Vector{Int})::Float64
    p_from = query_memory_system(from)
    p_to = query_memory_system(to)
    if p_from == 0. || p_to == 0.
        return 0.
    end
    from_future = from[2:length(from)]
    shortest = min(length(from_future), length(to))
    if from_future[1:shortest] == to[1:shortest]
        if length(from_future) >= length(to)
            return 1.
        else
            inter = vcat([from[1]], to)
            return query_memory_system(inter) / p_from
        end
    else
        return 0.
    end
end




