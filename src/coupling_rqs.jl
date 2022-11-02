# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

struct CouplingRQS <: Function
    nn1::Chain
    nn2::Chain
    mask1::AbstractVector
    mask2::AbstractVector
end

export CouplingRQS
@functor CouplingRQS

function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQS,
    x::AbstractMatrix{<:Real}
)
    return coupling_trafo(f, x)
end

(f::CouplingRQS)(x::AbstractMatrix{<:Real}) = coupling_trafo(f, x)[1]



function coupling_trafo(trafo::CouplingRQS, x::AbstractMatrix)

    # x₁ = reshape(x[trafo.mask1,:], length(trafo.mask1), size(x,2))
    # x₂ = reshape(x[trafo.mask2,:], length(trafo.mask2), size(x,2))

    # x₁ = view(x, trafo.mask1, axes(x,2))
    # x₂ = view(x, trafo.mask2, axes(x,2))

    x₁ = x[trafo.mask1, 1:end]
    x₂ = x[trafo.mask2, 1:end]

    y₁, LogJac₁ = partial_coupling_trafo(trafo.nn1, x₁, x₂)
    y₂, LogJac₂ = partial_coupling_trafo(trafo.nn2, x₂, y₁)

    return _sort_dimensions(y₁,y₂,trafo.mask1), LogJac₁ + LogJac₂
end

export coupling_trafo

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


#=
# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

struct CouplingRQSpline <: Function
    nn1::Chain
    nn2::Chain
end

function CouplingRQSpline(n_dims::Integer, K::Integer = 20)
    nn1, nn2 = _get_nns(n_dims, K)
    return CouplingRQSpline(nn1, nn2)
end

# function CouplingRQSpline(w1::AbstractMatrix, w2::AbstractMatrix, K::Integer = 20)

#     nn1 = Chain(Dense(w1, true, relu),
#                 Dense(20 => 20, relu),
#                 Dense(20 => (3K-1))
#                 )

#     nn2 = Chain(Dense(w2, true, relu),
#                 Dense(20 => 20, relu),
#                 Dense(20 => (3K-1))
#                 )
    
#     return CouplingRQSpline(nn1, nn2)
# end

export CouplingRQSpline
@functor CouplingRQSpline

Base.:(==)(a::CouplingRQSpline, b::CouplingRQSpline) = a.nn1 == b.nn1 &&  a.nn2 == b.nn1

Base.isequal(a::CouplingRQSpline, b::CouplingRQSpline) = isequal(a.nn1, b.nn1)  && isequal(a.nn2, b.nn2)

Base.hash(x::CouplingRQSpline, h::UInt) =  hash(x.nn1, hash(x.nn2, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h))))

(f::CouplingRQSpline)(x::AbstractMatrix{<:Real}) = coupling_trafo(f, x)[1]


struct CouplingRQSplineInv <: Function
    nn1::Chain
    nn2::Chain
end

export CouplingRQSplineInv
@functor CouplingRQSplineInv

Base.:(==)(a::CouplingRQSplineInv, b::CouplingRQSplineInv) = a.nn1 == b.nn1 &&  a.nn2 == b.nn1

Base.isequal(a::CouplingRQSplineInv, b::CouplingRQSplineInv) = isequal(a.nn1, b.nn1)  && isequal(a.nn2, b.nn2)

Base.hash(x::CouplingRQSplineInv, h::UInt) = hash(x.nn1, hash(x.nn2, hash(:TrainableRQSpline, hash(:EuclidianNormalizingFlows, h))))

(f::CouplingRQSplineInv)(x::AbstractMatrix{<:Real}) = coupling_trafo(f, x)[1]


function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQSpline,
    x::AbstractMatrix{<:Real}
)
    return coupling_trafo(f, x)
end

function InverseFunctions.inverse(f::CouplingRQSpline)
    return CouplingRQSplineInv(f.nn1, f.nn2)
end


function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQSplineInv,
    x::AbstractMatrix{<:Real}
)
    return coupling_trafo(f, x)
end

function InverseFunctions.inverse(f::CouplingRQSplineInv)
    return CouplingRQSpline(f.nn1, f.nn2)
end


function coupling_trafo(trafo::Union{CouplingRQSpline, CouplingRQSplineInv}, x::AbstractMatrix{<:Real})
    b = round(Int, size(x,1)/2)
    inv = trafo isa CouplingRQSplineInv 

    x₁ = x[1:b, :]
    x₂ = x[b+1:end, :]

    y₁, LogJac₁ = partial_coupling_trafo(trafo.nn1, x₁, x₂, inv)
    y₂, LogJac₂ = partial_coupling_trafo(trafo.nn2, x₂, y₁, inv)

    return vcat(y₁, y₂), LogJac₁ + LogJac₂
end

export coupling_trafo

function partial_coupling_trafo(nn::Chain, x₁::AbstractMatrix{<:Real}, x₂::AbstractMatrix{<:Real}, inv::Bool)
    N = size(x₁,2)

    θ = nn(x₂)
    K = Int((size(θ,1) + 1) / 3)
    w, h, d = get_params(θ, N, K)
    spline = inv ? RQSplineInv(w, h, d) : RQSpline(w, h, d)

    return with_logabsdet_jacobian(spline, x₁)
end

export partial_coupling_trafo


function _get_nns(n_dims::Integer, K::Integer = 20)

    d = n_dims > 2 ? round(Integer, n_dims / 2) : 1

    nn1 = Chain(Dense(d => 20, relu),
                Dense(20 => 20, relu),
                Dense(20 => (3K-1))
                )

    nn2 = Chain(Dense((n_dims - d) => 20, relu),
                Dense(20 => 20, relu),
                Dense(20 => (3K-1))
                )
    
    return nn1, nn2
end

function get_weights(n_dims::Integer)

    d = n_dims > 2 ? round(Integer, n_dims / 2) : 1

    w1 = Flux.glorot_uniform(20, d)
    w2 = Flux.glorot_uniform(20, d)

    return w1, w2
end

export get_weights

function get_params(θ::AbstractMatrix, N::Integer, K::Integer)

    w = _cumsum(_softmax(θ[1:K,:]))
    h = _cumsum(_softmax(θ[K+1:2K,:]))
    d = _softplus(θ[2K+1:end,:])

    w = vcat(repeat([-5,], 1, N), w)
    h = vcat(repeat([-5,], 1, N), h)
    d = vcat(repeat([1,], 1, N), d)
    d = vcat(d, repeat([1,], 1, N))

    return w, h, d
end

function _softmax(x::AbstractVector)
    exp_x = exp.(x)
    sum_exp_x = sum(exp_x)
    return exp_x ./ sum_exp_x 
end

function _softmax(x::AbstractMatrix)
    return hcat(_softmax.(x[:,i] for i in axes(x,2))...)
end

function _cumsum(x::AbstractVector; B = 5)
    return 2 .* B .* cumsum(x) .- B 
end

function _cumsum(x::AbstractMatrix; B = 5)
    return copy((2 .* B .* cumsum(x, dims = 1) .- B))
end

function _softplus(x::AbstractVector)
    return log.(exp.(x) .+ 1) 
end

function _softplus(x::AbstractMatrix)
    return (log.(exp.(x) .+ 1))
end

function _sigmoid(x::Real)
    return 1 / (1 + exp(-x))
end

function _relu(x::Real)
    return x > 0 ? x : 0
end


midpoint(lo::T, hi::T) where T<:Integer = lo + ((hi - lo) >>> 0x01)
binary_log(x::T) where {T<:Integer} = 8 * sizeof(T) - leading_zeros(x - 1)

function searchsortedfirst_impl(
        v::AbstractVector, 
        x::Real
    )
    
    u = one(Integer)
    lo = one(Integer) - u
    hi = length(v) + u
    
    n = binary_log(length(v))+1
    m = one(Integer)
    
    @inbounds for i in 1:n
        m_1 = midpoint(lo, hi)
        m = Base.ifelse(lo < hi - u, m_1, m)
        lo = Base.ifelse(v[m] < x, m, lo)
        hi = Base.ifelse(v[m] < x, hi, m)
    end
    return hi
end

=#