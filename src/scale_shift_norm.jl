# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

struct ScaleShiftNorm <: Function
    output_device
end

export ScaleShiftNorm
@functor ScaleShiftNorm

function ChangesOfVariables.with_logabsdet_jacobian(
    f::ScaleShiftNorm,
    x::Any
)
    return scale_shift_norm(x, f.output_device)
end

(f::ScaleShiftNorm)(x::Any) = scale_shift_norm(x, f.output_device)[1]
(f::ScaleShiftNorm)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::ScaleShiftNorm)
    throw(DomainError("Need to implement inverse of ScaleShiftNorm"))
end


struct ScaleShiftNormInverse <: Function
    output_device
end

export ScaleShiftNormInverse
@functor ScaleShiftNormInverse

function ChangesOfVariables.with_logabsdet_jacobian(
    f::ScaleShiftNormInverse,
    x::Any
)
    return scale_shift_norm(x, f.output_device)
end

(f::ScaleShiftNormInverse)(x::Any) = scale_shift_norm(x, f.output_device)[1]
(f::ScaleShiftNormInverse)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::ScaleShiftNormInverse)
    return ScaleShiftNorm(f.output_device)
end


function scale_shift_norm(x::AbstractMatrix, device)

    n_smpls = size(x,2)    
    stds = Float64[]
    y = deepcopy(x)
    for i in axes(x, 1)
        std_tmp  = std(y[i,:])
        y[i,:] .*= 1/std_tmp
        append!(stds, std_tmp)

        mean_tmp = mean(y[i,:])
        y[i,:] .-= mean_tmp
    end

    #calculate logabsdetjacobian to track volume change thats introduced by the scaling 
    ladj_scs = sum(log.(abs.(1 ./ stds)))
    ladj = fill(ladj_scs, 1, n_smpls)

    if device isa GPU
        y = gpu(y)
        ladj = gpu(ladj)
    end

    return y, ladj
end

export scale_shift_norm

