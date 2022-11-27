
export CouplingLayerGlow

struct CouplingLayerGlow <: NeuralNetLayer
    C::Conv1x1
    RB::Union{ResidualBlock, FluxBlock}
    logdet::Bool
    activation::ActivationFunction
end

@Flux.functor CouplingLayerGlow

# Constructor from input dimensions
function CouplingLayerGlow(n_in::Int64, n_hidden::Int64; freeze_conv=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, activation::ActivationFunction=SigmoidLayer(), ndims=2)

    # 1x1 Convolution and residual block for invertible layer
    C = Conv1x1(n_in; freeze=freeze_conv)
    RB = ResidualBlock(Int(n_in/2), n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=true, ndims=ndims)

    return CouplingLayerGlow(C, RB, logdet, activation)
end

# Forward pass: Input X, Output Y
function forward_conv_half_spline(X::AbstractArray{T, N}, L::CouplingLayerGlow) where {T,N}
    d = round(Int, size(X,1)/2)
    X_ = L.C(X)
    X1, X2 = X_[1:d,:], X_[d+1:end,:] 

    Y2 = copy(X2)
    θ = L.RB(X2)
    w, h, d = get_params(theta, size(X1, 1))
    Y1, ladj = with_logabsdet_jacobian(RQSpline(w,h,d),x)

    Y = vcat(Y1, Y2)

    return Y, ladj
end

# TODO: Inverse pass: Input Y, Output X
# function inverse(Y::AbstractArray{T, N}, L::CouplingLayerGlow; save=false) where {T,N}
#     Y1, Y2 = tensor_split(Y)

#     X2 = copy(Y2)
#     logS_T = L.RB.forward(X2)
#     logSm, Tm = tensor_split(logS_T)
#     Sm = L.activation.forward(logSm)
#     X1 = (Y1 - Tm) ./ (Sm .+ eps(T)) # add epsilon to avoid division by 0

#     X_ = tensor_cat(X1, X2)
#     X = L.C.inverse(X_)

#     save == true ? (return X, X1, X2, Sm) : (return X)
# end
