export ActNorm, reset!

mutable struct ActNorm <: NeuralNetLayer
    k::Integer
    s::AbstractVector
    b::AbstractVector
    is_reversed::Bool
end

@functor ActNorm

# Constructor: Initialize with nothing
function ActNorm(k)
    s = nothing
    b = nothing
    return ActNorm(k, s, b, false)
end

# 2D Foward pass: Input X, Output Y
function forward(X::AbstractMatrix{T}, AN::ActNorm) where T
    # Initialize during first pass such that
    # output has zero mean and unit variance
    if isnothing(AN.s) && !AN.is_reversed
        μ = mean(X; dims=2)
        σ_sqr = var(X; dims=2)
        AN.s = 1 ./ sqrt.(σ_sqr)
        AN.b = -μ ./ sqrt.(σ_sqr)
    end
    Y = X .* AN.s .+ AN.b
    logdet = fill(sum(log.(abs.(AN.s))), 1, size(X,2))

    return Y, logdet
end

# 2-3D Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, N}, AN::ActNorm; logdet=nothing) where {T, N}
    isnothing(logdet) ? logdet = (AN.logdet && AN.is_reversed) : logdet = logdet
    inds = [i!=(N-1) ? 1 : Colon() for i=1:N]
    dims = collect(1:N-1); dims[end] +=1

    # Initialize during first pass such that
    # output has zero mean and unit variance
    if isnothing(AN.s.data) && AN.is_reversed
        μ = mean(Y; dims=dims)[inds...]
        σ_sqr = var(Y; dims=dims)[inds...]
        AN.s.data = sqrt.(σ_sqr)
        AN.b.data = μ
    end
    X = (Y .- reshape(AN.b.data, inds...)) ./ reshape(AN.s.data, inds...)

    # If logdet true, return as second ouput argument
    logdet ? (return X, -logdet_forward(size(Y)[1:N-2]..., AN.s)) : (return X)
end

## Logdet utils
# 2D Logdet
logdet_forward(nx, ny, s) = nx*ny*sum(log.(abs.(s.data)))
logdet_backward(nx, ny, s) = nx*ny ./ s.data
logdet_hessian(nx, ny, s) = -nx*ny ./ s.data.^2f0

# Reverse
function tag_as_reversed!(AN::ActNorm, tag::Bool)
    AN.is_reversed = tag
    return AN
end
