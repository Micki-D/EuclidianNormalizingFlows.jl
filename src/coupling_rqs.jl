# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

struct CouplingRQSBlock <: Function
    nn::Chain
    mask::AbstractVector
end

export CouplingRQSBlock
@functor CouplingRQSBlock

function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQSBlock,
    x::Any
)
    return rqs_trafo!(f, x)
end

(f::CouplingRQSBlock)(x::Any) = rqs_trafo!(f, x)[1]
(f::CouplingRQSBlock)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::CouplingRQSBlock)
    return CouplingRQSBlockInverse(f.nn, f.mask)
end


struct CouplingRQSBlockInverse <: Function
    nn::Chain
    mask::AbstractVector
end

export CouplingRQSBlockInverse
@functor CouplingRQSBlockInverse

function ChangesOfVariables.with_logabsdet_jacobian(
    f::CouplingRQSBlockInverse,
    x::Any
)
    return rqs_trafo!(f, x)
end

(f::CouplingRQSBlockInverse)(x::Any) = rqs_trafo!(f, x)[1]
(f::CouplingRQSBlockInverse)(vs::AbstractValueShape) = vs

function InverseFunctions.inverse(f::CouplingRQSBlockInverse)
    return CouplingRQSBlock(f.nn, f.mask)
end


function rqs_trafo!(trafo::Union{CouplingRQSBlock, CouplingRQSBlockInverse}, x::Any)

    spline = trafo isa CouplingRQSBlock ? RQSpline : RQSplineInv
    x[trafo.mask,:], ladj = with_logabsdet_jacobian(spline(trafo.nn(x[.~trafo.mask,:])), x[trafo.mask,:])

    return ladj
end

export rqs_trafo!
