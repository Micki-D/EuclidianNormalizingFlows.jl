export ActNorm, ActNormInv, reset!

mutable struct ActNorm <: Function
    k::Integer
    s::AbstractVector
    b::AbstractVector
    is_reversed::Bool
end

function ActNorm(k)
    s = Float64[]
    b = Float64[]
    return ActNorm(k, s, b, false)

end

@functor ActNorm

function ChangesOfVariables.with_logabsdet_jacobian(
    f::ActNorm,
    x::AbstractMatrix{<:Real}
)
    return forward(f, x)
end

(f::ActNorm)(x::AbstractMatrix) = forward(f,x)[1]

function InverseFunctions.inverse(f::ActNorm)
    return ActNormInv(f.k, f.s, f.b, f.is_reversed)
end

mutable struct ActNormInv <: Function
    k::Integer
    s::AbstractVector
    b::AbstractVector
    is_reversed::Bool
end

@functor ActNormInv

function ChangesOfVariables.with_logabsdet_jacobian(
    f::ActNormInv,
    x::AbstractMatrix{<:Real}
)
    return inverse(f, x)
end

(f::ActNormInv)(x::AbstractMatrix) = inverse(f,x)[1]

function InverseFunctions.inverse(f::ActNormInv)
    return ActNorm(f.k, f.s, f.b, f.is_reversed)
end



# Foward pass: Input X, Output Y
function forward(AN::ActNorm, X::AbstractMatrix{T}) where T
    # Initialize during first pass such that
    # output has zero mean and unit variance
    if isempty(AN.s)
        μ = vec(mean(X; dims=2))
        σ_sqr = vec(var(X; dims=2))
        AN.s = 1 ./ sqrt.(σ_sqr)
        AN.b = -μ ./ sqrt.(σ_sqr)
    end

    Y = X .* AN.s .+ AN.b
    logdet = fill(sum(log.(abs.(AN.s))), 1, size(X,2))

    return Y, logdet
end

# Inverse pass: Input Y, Output X
function inverse(AN::ActNormInv, Y::AbstractMatrix{T}) where T
    # Initialize during first pass such that
    # output has zero mean and unit variance
    if isempty(AN.s)
        μ = vec(mean(Y; dims=2))
        σ_sqr = vec(var(Y; dims=2))
        AN.s = sqrt.(σ_sqr)
        AN.b = μ
    end
    
    X = (Y .- AN.b) ./ AN.s
    logdet = fill(-sum(log.(abs.(AN.s))), 1, size(X,2))

    return X, logdet
end

# Reverse
function tag_as_reversed!(AN::ActNorm, tag::Bool)
    AN.is_reversed = tag
    return AN
end
