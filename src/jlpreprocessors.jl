module JLPreprocessors

include("MultivariateStats/src/MultivariateStats.jl")
using .MultivariateStats

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!

export JLPreprocessor

const jlpreproc_dict = Dict(
	"PCA" => PCA, 
	"PPCA" => PPCA,
	"ICA" => ICA, 
	"FA" => FactorAnalysis,
)

mutable struct JLPreprocessor <: Learner
  name::String
  model::Dict
  args::Dict
  
  function JLPreprocessor(args=Dict())
    default_args = Dict( 
      :name => "jlprep",
		:preprocessor => "PCA",
      :impl_args => Dict(),
		:autocomponent => true,
		:n_components =>0
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
  autocomp = prep.args[:autocomponent]
  ncomponents=prep.args[:n_components]
  xn = (Matrix(x))' |> collect
  if ncomponents == 0
	 ncomponents = min(size(xn)...)
	 impl_args[:n_components] = ncomponents
  end
  if autocomp == true
	 cols = ncol(x)
	 if cols > 0
		ncomponents = round(sqrt(cols),digits=0) |> Integer
		impl_args[:n_components] = ncomponents
	 end
  end
  preproc = nothing
  if proc == ICA
	 preproc = fit(proc,xn,ncomponents,tol=0.05)
  else
	 preproc = fit(proc,xn,maxoutdim = ncomponents)
  end
  prep.model = Dict(
		:preprocessor => preproc,
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
