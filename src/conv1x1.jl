
export Conv1x1

struct Conv1x1 <: NeuralNetLayer
    k::Integer
    v1::AbstractVector
    v2::AbstractVector
    v3::AbstractVector
    freeze::Bool
end

@Flux.functor Conv1x1

function Conv1x1(v1, v2, v3; freeze=false)
    k = length(v1)
    return Conv1x1(k, v1, v2, v3, freeze)
end

(c::Conv1x1)(x::AbstractArray) = forward_conv1x1(x,c)

# Forward pass
function forward_conv1x1(X::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    Y = cuzeros(X, size(X)...)
    n_in = size(X, N-1)

    for i=1:size(X, N)
        Xi = reshape(selectdim(X, N, i), :, n_in)
        Yi = chain_lr(Xi, C.v1, C.v2, C.v3)
        selectdim(Y, N, i) .= reshape(Yi, size(selectdim(Y, N, i))...)
    end

    return Y # logdet always 0
end


# Inverse pass
function inverse(Y::AbstractArray{T, N}, C::Conv1x1) where {T, N}
    X = cuzeros(Y, size(Y)...)
    n_in = size(X, N-1)

    for i=1:size(Y, N)
        Yi = reshape(selectdim(Y, N, i), :, n_in)
        Xi = chain_lr(Yi, C.v1, C.v2, C.v3)
        selectdim(X, N, i) .= reshape(Xi, size(selectdim(X, N, i))...)
    end

    return X # logdet always 0
end


function inverse(C::Conv1x1)
    return Conv1x1(C.k, C.v3, C.v2, C.v1, C.logdet, C.freeze)
end

# for 1x1 Conv
gemm_outer!(out::Matrix{T}, tmp::Vector{T}, v::Vector{T}) where T = LinearAlgebra.BLAS.gemm!('N', 'T', T(1), tmp, v, T(1), out)
gemm_outer!(out::CuMatrix{T}, tmp::CuVector{T}, v::CuVector{T}) where T = CUDA.CUBLAS.gemm!('N', 'T', T(1), tmp, v, T(1), out)

function chain_lr(x::AbstractMatrix{T}, vi::Vararg{AbstractVector{T}, N}) where {T, N}
    out = T(1) .* x
    tmp = cuzeros(vi[1], size(x, 1))
    for v=vi
        n = -2/norm(v)^2
        mul!(tmp, out, v)
        rmul!(tmp, n)
        gemm_outer!(out, tmp, v)
    end
    out
end
