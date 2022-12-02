using Revise 
using EuclidianNormalizingFlows

using BAT
using Distributions
using Optimisers
using FunctionChains
using ArraysOfArrays
using LinearAlgebra
using ValueShapes
using QuadGK
using StatsBase
using FileIO
using JLD2
using CUDA
using CUDAKernels
using KernelAbstractions
using Flux
using PyPlot

function uniformity_measure(samples::AbstractVector{<:Real}) ## samples are assumed to be uniform, since you compare against a Uniform distribution
    sort!(samples)
    n = length(samples)
    gcdf = ecdf(samples)
    f(x::Real) = (gcdf(x) - x )^2
    res = quadgk(f, 0, samples[1])[1]
    for i in 2:n
        res += quadgk(f, samples[i-1], samples[i])[1]
    end
    res += quadgk(f, samples[end], 1)[1]
    return res
end

function get_scale_shifted_samples(n_dims::Integer, nsamples::Integer, device)
    d = MixtureModel(MvNormal[
        MvNormal(randn(n_dims), 0.1 .* I(n_dims)),
        MvNormal(randn(n_dims), 0.04 .* I(n_dims)),
        MvNormal(randn(n_dims), 0.04 .* I(n_dims)),
        MvNormal(randn(n_dims), 0.3 .* I(n_dims)),
        MvNormal(randn(n_dims), I(n_dims))])


    samples = bat_sample(d, 
                    BAT.IIDSampling(nsamples=n_smpls)
                    ).result;


    smpls_flat = ValueShapes.flatview(unshaped.(samples.v))  

    #scale and shift to fit inside [-5,5] interval in each dimension
    stds = Float64[]
    for i in axes(smpls_flat, 1)
    std_tmp  = std(smpls_flat[i,:])
    smpls_flat[i,:] .*= 1/std_tmp
    append!(stds, std_tmp)

    mean_tmp = mean(smpls_flat[i,:])
    smpls_flat[i,:] .-= mean_tmp
    end

    #calculate logabsdetjacobian to track volume change thats introduced by the scaling 
    ladj_scs = sum(log.(abs.(1 ./ stds)))
    ladj1 = fill(ladj_scs, 1, n_smpls)

    samples_flat = ValueShapes.flatview(unshaped.(samples.v))
    if device isa GPU
        samples_flat = gpu(samples_flat)
        ladj1 = gpu(ladj1)
    end

    return samples, samples_flat, ladj1
end 


wanna_use_GPU = false
_device = wanna_use_GPU ? KernelAbstractions.get_device(CUDA.rand(10)) : KernelAbstractions.get_device(rand(10))

integrals = []
stepsizes = []
batches = []
nepochs = []


n_smpls = 5 * 10^3
n_dims = 4
importance_density = MvNormal(zeros(n_dims), I(n_dims)) 

smpls1_dsv, smpls_flat, ladj1 = get_scale_shifted_samples(n_dims, n_smpls, _device);
smpls= nestedview(smpls_flat);



nbatches = 80
nepochs = 30

ls = 1f-3
lm = 1f-2
le = 1f-5

phase_durations = [1/nepochs, 1/nepochs, (nepochs-3)/nepochs, 1/nepochs ]

K = 7

flow = get_flow(n_dims,_device,K)


r = optimize_whitening_annealing(smpls1_dsv, flow, nbatches, nepochs, ls, lm, le, phase_durations)
