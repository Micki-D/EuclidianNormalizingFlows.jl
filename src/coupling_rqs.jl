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






struct HardFlow <: Function
    C::Conv1x1
    RQS1::CouplingRQS
    RQS2::CouplingRQS
    freeze_conv::Bool
end

@functor HardFlow

function ChangesOfVariables.with_logabsdet_jacobian(
    f::HardFlow,
    x::AbstractMatrix{<:Real}
)
    return forward(f, x)
end

(f::HardFlow)(x::AbstractMatrix{<:Real}) = forward(f, x)[1]







function forward(trafo::HardFlow, input::AbstractMatrix, freeze_conv::Bool=false)

    X = freeze_conv ? input : trafo.C(input)

    μ = vec(mean(X; dims=2))
    σ_sqr = vec(var(X; dims=2))
    σ_inv = 1 ./ sqrt.(σ_sqr)
    x = (X .- μ) .* σ_inv
    logdet_scale_shift1 = fill(sum(log.(abs.(σ_inv))), 1, size(X,2))
    d = ceil(Int, size(x,1)/2)

    y, logdet_rqs1 = EuclidianNormalizingFlows.with_logabsdet_jacobian(trafo.RQS1,x)

    μ2 = vec(mean(y; dims=2))
    σ_sqr2 = vec(var(y; dims=2))
    σ_inv2 = 1 ./ sqrt.(σ_sqr2)
    y₁ = (y .- μ2) .* σ_inv2
    logdet_scale_shift2 = fill(sum(log.(abs.(σ_inv2))), 1, size(X,2))

    y₂, logdet_rqs2 =  EuclidianNormalizingFlows.with_logabsdet_jacobian(trafo.RQS1,vcat(y₁[d+1:end, :], y₁[1:d, :]))


    output = freeze_conv ? vcat(y₂[1:d, :], y₂[d+1:end, :]) : InverseFunctions.inverse(trafo.C)(vcat(y₂[1:d, :], y₂[d+1:end, :]))
    #output = vcat(y₂[1:d, :], y₂[d+1:end, :])

    return output, logdet_scale_shift1 + logdet_scale_shift2 + logdet_rqs1 + logdet_rqs2
end



function forward(trafo::CouplingRQS, X::AbstractMatrix)

    μ = vec(mean(X; dims=2))
    σ_sqr = vec(var(X; dims=2))
    σ_inv = 1 ./ sqrt.(σ_sqr)

    x = (X .- μ) .* σ_inv
    logdet_scale_shift = fill(sum(log.(abs.(σ_inv))), 1, size(X,2))

    d = ceil(Int, size(x,1)/2)
    x₁ = CUDA.@allowscalar(x[1:d, :])
    x₂ = CUDA.@allowscalar(x[d+1:end, :])

    θ = trafo.nn(x₂)
    w, h, d = get_params(θ, size(x₁,1))
    spline = RQSpline(w, h, d)

    y₁, LogJac =  with_logabsdet_jacobian(spline, x₁)

    return vcat(y₁,x₂), LogJac + logdet_scale_shift
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

function partial_coupling_trafo(nn::Chain, 
    x₁::AbstractMatrix{<:Real}, 
    x₂::AbstractMatrix{<:Real}
)
    θ = nn(x₂)
    w, h, d = get_params(θ, size(x₁,1))
    spline = RQSpline(w, h, d)

    return with_logabsdet_jacobian(spline, x₁)
end

export partial_coupling_trafo
