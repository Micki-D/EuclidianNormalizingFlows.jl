using Pkg
Pkg.activate("/home/micki/.julia/environments/splines")
using BAT
using ValueShapes
using Distributions
using PyPlot
using Optimisers
using ArraysOfArrays
using FunctionChains
#using Optim
#using DensityRatioEstimation
using InverseFunctions
using LinearAlgebra
#using FileIO

using Revise
using EuclidianNormalizingFlows


n_smpls = 5 * 10^3
reps = 10

dim_list = vcat(fill(10, reps), fill(20,reps))
integrals = Real[]

for n_dims in dim_list 
    # define test density 
    d = MixtureModel(MvNormal[
            MvNormal(randn(n_dims), 0.1 .* I(n_dims)),
            MvNormal(randn(n_dims), 0.04 .* I(n_dims)),
            MvNormal(randn(n_dims), 0.04 .* I(n_dims)),
            MvNormal(randn(n_dims), 0.3 .* I(n_dims)),
            MvNormal(randn(n_dims), I(n_dims))])


    #draw samples 
    samples = bat_sample(d, 
                        BAT.IIDSampling(nsamples=n_smpls)
                        ).result


    smpls_flat = ValueShapes.flatview(unshaped.(samples.v))  


    # normalize samples, so that they lie in the [-5,5] interval mask required for the rqs transformation

    stds = Float64[]

    for i in axes(smpls_flat, 1)
        std_tmp  = std(smpls_flat[i,:])
        smpls_flat[i,:] .*= 1/std_tmp
        append!(stds, std_tmp)
        
        mean_tmp = mean(smpls_flat[i,:])
        smpls_flat[i,:] .-= mean_tmp
    end

    # store ladj as introduced by the scaling operation
    ladj_scs = sum(log.(abs.(1 ./ stds)))
    ladj1 = fill(ladj_scs, 1, n_smpls)


    # (optional) check for outliers
    outliers = broadcast(x -> abs(x) >= 5.0, smpls_flat)
    ids = Int[]
    for i in 1:n_smpls
        if any(outliers[:,i])
            append!(ids, i)
        end
    end

    # get one block of transformations
    initial_trafo = get_flow(n_dims,15).fs[1]

    # training parameters 
    nbatches = 30
    nepochs = 150
    shuffle_samples = true
    stepsize = 3f-4

    optimizer = Optimisers.Adam(stepsize)
    smpls = nestedview(smpls_flat)

    # train trafo
    r = EuclidianNormalizingFlows.optimize_whitening(smpls, 
        initial_trafo, 
        optimizer,
        nbatches = nbatches,
        nepochs = nepochs,
        shuffle_samples = shuffle_samples)

    trained_trafo = r.result

    smpls_transformed, ladj2 = EuclidianNormalizingFlows.with_logabsdet_jacobian(trained_trafo, smpls_flat)

    # integrate via generalized harmonic mean estimator 
    importance_density = MvNormal(zeros(n_dims), I(n_dims)) 
    ladj_trafo = ladj1 + ladj2

    log_posterior = samples.logd - vec(ladj_trafo)
    frac = [logpdf(importance_density, smpls_transformed[:,i]) / log_posterior[i] for i in 1:n_smpls]

    integral = n_smpls * inv(sum(frac)) 

    append!(integrals, integral)

    println("Training in $n_dims dimensions now.")
end

nbatches = 30
nepochs = 150
shuffle_samples = true
stepsize = 3f-4

int_10 = integrals[1:reps]
int_20 = integrals[reps+1:2*reps]
#int_30 = integrals[2*reps+1:end]

save("nice_benchmark_data.jld2", Dict(
    "int_10D" => int_10,
    "int_20D" => int_20,

    "dim_list" => dim_list, 
    "integrals" => integrals,

    "config" => Dict("dist" => "Gaussian Mixture", "nbatches" => nbatches, "nepochs" => nepochs, "stepsize" => stepsize)

))

