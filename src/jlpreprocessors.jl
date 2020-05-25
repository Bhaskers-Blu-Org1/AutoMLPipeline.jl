module JLPreprocessors

#include("MultivariateStats/src/MultivariateStats.jl")
#using .MultivariateStats: fit, transform, PCA, PPCA, ICA, FactorAnalysis

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!

export JLPreprocessor

function __init__()
  @eval using MultivariateStats
  global jlpreproc_dict = Dict(
										"PCA" => PCA, 
										"PPCA" => PPCA,
										"ICA" => ICA, 
										"FA" => FactorAnalysis,
										"Whiten" => Whitening
										)
end


mutable struct JLPreprocessor <: Learner
  name::String
  model::Dict
  args::Dict
  
  function JLPreprocessor(args=Dict())
    default_args = Dict( 
      :name         => "pca",
		:preprocessor => "PCA",
		:tol          => 1.0,
		:n_components => 0,
      :impl_args    => Dict()
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
	 prep = cargs[:preprocessor]
	 if !(prep in keys(jlpreproc_dict))
      println("$prep is not supported.")
      println()
		println("JLPreprocessors: ", keys(jlpreproc_dict))
      error("Argument keyword error")
    end
    new(cargs[:name],Dict(),cargs)
  end
end

function JLPreprocessor(prep::String, args::Dict=Dict())
  JLPreprocessor(Dict(:preprocessor=>prep,:name=>prep,args...))
end

function fit!(prep::JLPreprocessor, x::DataFrame, y::Vector=[])
  @assert !isempty(x)
  impl_args = copy(prep.args[:impl_args])
  proc = jlpreproc_dict[prep.args[:preprocessor]]
  ncomponents=prep.args[:n_components]
  xn = (Matrix(x))' |> collect
  # if ncomponents not set, use autocomp
  maxcomp = min(size(xn)...)
  if ncomponents == 0
	 # use autocomp
	 nc = round(sqrt(maxcomp),digits=0) |> Integer
	 impl_args[:n_components] = nc
  else # autocomp
	 impl_args[:n_components] = ncomponents
  end
  pmodel = nothing
  if proc == ICA
	 tolerance = prep.args[:tol]
	 pmodel = fit(proc,xn,ncomponents,tol=tolerance)
  else
	 pmodel = fit(proc,xn,maxoutdim = ncomponents)
  end
  prep.model = Dict(
						  :preprocessor => pmodel,
						  :impl_args => impl_args
						 )
end

function transform!(prep::JLPreprocessor, x::DataFrame)
  xn = (Matrix(x))' |> collect
  preproc = prep.model[:preprocessor]
  res=transform(preproc,xn)
  return res' |> collect |> DataFrame
end

end # module
