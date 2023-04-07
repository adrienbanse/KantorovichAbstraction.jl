abstract type AbstractDynamicalSystem end 

mutable struct DynamicalSystem{FType<:Function, HType<:Function, OracleType<:Function} <:AbstractDynamicalSystem
    F::FType
    H::HType
    dim::Int
    state::Vector{Float64}
    outputs::Vector{Int}
    output::Int
    oracle::OracleType
end
F(sys::DynamicalSystem) = sys.F
H(sys::DynamicalSystem) = sys.H
dim(sys::DynamicalSystem) = sys.dim
state(sys::DynamicalSystem) = sys.state
outputs(sys::DynamicalSystem) = sys.outputs
output(sys::DynamicalSystem) = sys.output
oracle(sys::DynamicalSystem) = sys.oracle
function next!(sys::DynamicalSystem)
    sys.state = (F(sys))(state(sys))
    sys.output = (H(sys))(state(sys))
end

mutable struct ControlledDynamicalSystem{KType<:Function} <:AbstractDynamicalSystem
    system::DynamicalSystem
    K::KType
end
F(sys::ControlledDynamicalSystem) = F(sys.system)
H(sys::ControlledDynamicalSystem) = H(sys.system)
dim(sys::ControlledDynamicalSystem) = dim(sys.system)
state(sys::ControlledDynamicalSystem) = state(sys.system)
outputs(sys::ControlledDynamicalSystem) = outputs(sys.system)
output(sys::ControlledDynamicalSystem) = output(sys.system)
oracle(sys::ControlledDynamicalSystem) = oracle(sys.system)
K(sys::ControlledDynamicalSystem) = sys.K
function next!(sys::ControlledDynamicalSystem)
    sys.system.state = (F(sys))((K(sys))(state(sys)))
    sys.system.output = (H(sys))(state(sys))
end


