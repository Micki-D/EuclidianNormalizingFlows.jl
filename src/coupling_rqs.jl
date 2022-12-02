# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

struct CouplingRQS <: Function
    nn::Chain
end

export CouplingRQS
@functor CouplingRQS

struct PRQS <: Function
    nn::Chain
    params::AbstractArray
end 

export PRQS
@functor PRQS

function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQS,
    x::AbstractMatrix{<:Real}
)
    return forward(f, x)
end

(f::CouplingRQS)(x::AbstractMatrix{<:Real}) = forward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::PRQS,
    x::AbstractMatrix{<:Real}
)
    return indie_trafo(f, x)
end

(f::PRQS)(x::AbstractMatrix{<:Real}) = indie_trafo(f, x)[1]


function forward(trafo::CouplingRQS, x::AbstractMatrix)
    d = ceil(Int, size(x,1)/2)

    x₁ = CUDA.@allowscalar(x[1:d, :])
    x₂ = CUDA.@allowscalar(x[d+1:end, :])

    θ = trafo.nn(x₂)
    w, h, d = get_params(θ, size(x₁,1))
    spline = RQSpline(w, h, d)

    y₁, LogJac =  with_logabsdet_jacobian(spline, x₁)

    return vcat(y₁,x₂), LogJac
end

export coupling_trafo

function indie_trafo(trafo::PRQS, x::AbstractMatrix)

    d = ceil(Int, size(x,1)/2)

    x₁ = CUDA.@allowscalar(x[1:d, :])
    x₂ = CUDA.@allowscalar(x[d+1:end, :])

    w, h, d = get_params(trafo.params, size(x₁,1))
    w = gpu(w)
    h = gpu(h)
    d = gpu(d)

    spline = RQSpline(w, h, d)

    y₁, LogJac₁ = with_logabsdet_jacobian(spline, x₁)
    y₂, LogJac₂ = partial_coupling_trafo(trafo.nn2, x₂, y₁)

    return _sort_dimensions(y₁,y₂,trafo.mask1), LogJac₁ + LogJac₂
end
