module JLPreprocessors

#include("MultivariateStats/src/MultivariateStats.jl")
#using .MultivariateStats: fit, transform, PCA, PPCA, ICA, FactorAnalysis

# standard included modules
using DataFrames: DataFrame, nrow, ncol
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils
using AutoMLPipeline

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!

export JLPreprocessor
export testjlprep

# need this because multivariatestats load binaries thru arpack
function __init__()
  @eval using MultivariateStats
  global MV = MultivariateStats
  global jlpreproc_dict = Dict(
										"PCA" => PCA, 
										"PPCA" => PPCA,
										"ICA" => ICA, 
										"FA" => FactorAnalysis
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
  pmodel = nothing
  if ncomponents == 0
	 # use autocomp
	 if proc == ICA 
		maxcomp = ncol(x)
		ncomponents = round(sqrt(maxcomp),digits=0) |> Integer
		tolerance = prep.args[:tol]
		pmodel = MV.fit(proc,xn,ncomponents,tol=tolerance)
	 elseif proc == PCA || proc == PPCA
		maxcomp = min(size(xn)...)
		ncomponents = round(sqrt(maxcomp),digits=0) |> Integer
		pmodel = MV.fit(proc,xn,maxoutdim = ncomponents)
	 elseif proc == FactorAnalysis
		maxcomp = ncol(x)-1
		ncomponents = round(sqrt(maxcomp),digits=0) |> Integer
		pmodel = MV.fit(proc,xn,maxoutdim = ncomponents)
	 end
  else
	 if proc == ICA
		tolerance = prep.args[:tol]
		pmodel = MV.fit(proc,xn,ncomponents,tol=tolerance)
	 else
		pmodel = MV.fit(proc,xn,maxoutdim = ncomponents)
	 end
  end
  impl_args[:n_components] = ncomponents
  prep.model = Dict(
						  :preprocessor => pmodel,
						  :impl_args => impl_args
						 )
end

function transform!(prep::JLPreprocessor, x::DataFrame)
  xn = (Matrix(x))' |> collect
  preproc = prep.model[:preprocessor]
  res=MV.transform(preproc,xn)
  return res' |> collect |> DataFrame
end

function testjlprep(nc::Int)
  profbdata = getprofb()
  X = profbdata[:,2:end]
  Y = profbdata[:,1] |> Vector;
  ohe = OneHotEncoder()
  catf = CatFeatureSelector();
  numf = NumFeatureSelector()
  rf = RandomForest();
  ada = Adaboost()
  dt=PrunedTree()
  pca = JLPreprocessor("PCA",Dict(:n_components => nc))
  fa = JLPreprocessor("FA",Dict(:n_components   => nc))
  ica = JLPreprocessor("ICA",Dict(:tol          => 1.0,:n_components => nc))
  accuracy(X,Y)=score(:accuracy,X,Y)
  acc=[]
  learners=[rf,ada,dt]
  for lr in learners
	 println(lr.name)
	 #pipe=@pipeline ((catf |> ohe) +((numf) |> (fa +  ica + pca ))) |> lr
	 pipe=@pipeline ((catf |> ohe) +((numf) |> ( ica + pca ))) |> lr
	 m=crossvalidate(pipe,X,Y,accuracy,10,false)
	 push!(acc,m)
	 println(m)
  end
end

end # module
