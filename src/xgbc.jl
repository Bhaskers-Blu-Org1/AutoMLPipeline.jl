module XGBoostLearners

import XGBoost
import MLBase

# standard modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!

export Xgbc

mutable struct Xgbc <: Learner
  name::String
  model::Dict
  args::Dict

  function Xgbc(args::Dict = Dict())
	 default_args = Dict(
								:name => "xgbc",
								:impl_args => Dict(),
								:num_round => 100,
								:silent => true
								)
	 cargs=nested_dict_merge(default_args,args)
	 cargs[:name] = cargs[:name]*"_"*randstring(3)
	 new(cargs[:name],Dict(),cargs)
  end
end

function fit!(xgb::Xgbc, adf::DataFrame, labels::Vector)
  @assert !isempty(adf)
  @assert !isempty(labels)
  label_map = MLBase.labelmap(labels)
  ilab = MLBase.labelencode(label_map,labels) .- 1
  lab32 = convert(Array{Int32}, ilab)
  df32 = convert(Array{Float32}, adf)
  dtrain = XGBoost.DMatrix(df32,label=lab32)
  impl_args = xgb.args[:impl_args]
  num_round = xgb.args[:num_round]
  param = ["objective"=>"multi:softmax","num_class"=>length(label_map), impl_args...]
  silence = xgb.args[:silent]
  boostm = XGBoost.xgboost(dtrain,num_round, param=param,silent=silence)
  xgb.model = Dict(
   :label_map => label_map,
   :xgbmodel => boostm
  )
end

function transform!(xgb::Xgbc, adf::DataFrame)
  @assert !isempty(adf)
  boostm = xgb.model[:xgbmodel]
  df = deepcopy(adf) 
  df32 = convert(Array{Float32}, df)
  pred = XGBoost.predict(boostm,df32) .|> Int
  label_map = xgb.model[:label_map]
  res=MLBase.labeldecode(label_map,pred .+ 1) 
  return res 
end

end
