export NetworkGlow

struct NetworkGlow <: InvertibleNetwork
    AN::AbstractArray{ActNorm, 2}
    CL::AbstractArray{CouplingLayerGlow, 2}
    Z_dims::Union{Array{Array, 1}, Nothing}
    L::Int64
    K::Int64
    squeezer::Squeezer
    split_scales::Bool
end

@Flux.functor NetworkGlow

# Constructor
function NetworkGlow(n_in, n_hidden, L, K; freeze_conv=false, split_scales=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2, squeezer::Squeezer=ShuffleLayer(), activation::ActivationFunction=SigmoidLayer())
    AN = Array{ActNorm}(undef, L, K)    # activation normalization
    CL = Array{CouplingLayerGlow}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
 
    if split_scales
        Z_dims = fill!(Array{Array}(undef, L-1), [1,1]) #fill in with dummy values so that |> gpu accepts it   # save dimensions for inverse/backward pass
        channel_factor = 2^(ndims)
    else
        Z_dims = nothing
        channel_factor = 1
    end

    for i=1:L
        n_in *= channel_factor # squeeze if split_scales is turned on
        for j=1:K
            AN[i, j] = ActNorm(n_in; logdet=true)
            CL[i, j] = CouplingLayerGlow(n_in, n_hidden; freeze_conv=freeze_conv, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, activation=activation, ndims=ndims)
        end
        (i < L && split_scales) && (n_in = Int64(n_in/2)) # split
    end

    return NetworkGlow(AN, CL, Z_dims, L, K, squeezer, split_scales)
end

# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, G::NetworkGlow) where {T, N}
    G.split_scales && (Z_save = array_of_array(X, G.L-1))

    logdet = 0
    for i=1:G.L
        (G.split_scales) && (X = G.squeezer.forward(X))
        for j=1:G.K            
            X, logdet1 = G.AN[i, j].forward(X)
            X, logdet2 = G.CL[i, j].forward(X)
            logdet += (logdet1 + logdet2)
        end
        if G.split_scales && i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = collect(size(Z))
        end
    end
    G.split_scales && (X = cat_states(Z_save, X))
    return X, logdet
end

# Inverse pass 
function inverse(X::AbstractArray{T, N}, G::NetworkGlow) where {T, N}
    G.split_scales && ((Z_save, X) = split_states(X, G.Z_dims))
    for i=G.L:-1:1
        if G.split_scales && i < G.L
            X = tensor_cat(X, Z_save[i])
        end
        for j=G.K:-1:1
            X = G.CL[i, j].inverse(X)
            X = G.AN[i, j].inverse(X)
        end

        (G.split_scales) && (X = G.squeezer.inverse(X))
    end
    return X
end
