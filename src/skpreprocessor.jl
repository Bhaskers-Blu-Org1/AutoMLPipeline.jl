module SKPreprocessors

using PyCall

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export SKPreprocessor, skpreprocessors


function __init__()
  global gDEC  = pyimport_conda("sklearn.decomposition","scikit-learn")
  global gFS   = pyimport_conda("sklearn.feature_selection","scikit-learn")
  global gIMP  = pyimport_conda("sklearn.impute","scikit-learn")
  global gPREP = pyimport_conda("sklearn.preprocessing","scikit-learn")

  # Available scikit-learn learners.
  global preprocessor_dict = Dict(
     "DictionaryLearning"          => gDEC.DictionaryLearning,
     "FactorAnalysis"              => gDEC.FactorAnalysis,
     "FastICA"                     => gDEC.FastICA,
     "IncrementalPCA"              => gDEC.IncrementalPCA,
     "KernelPCA"                   => gDEC.KernelPCA,
     "LatentDirichletAllocation"   => gDEC.LatentDirichletAllocation,
     "MiniBatchDictionaryLearning" => gDEC.MiniBatchDictionaryLearning,
     "MiniBatchSparsePCA"          => gDEC.MiniBatchSparsePCA,
     "NMF"                         => gDEC.NMF,
     "PCA"                         => gDEC.PCA,
     "SparsePCA"                   => gDEC.SparsePCA,
     "SparseCoder"                 => gDEC.SparseCoder,
     "TruncatedSVD"                => gDEC.TruncatedSVD,
     "dict_learning"               => gDEC.dict_learning,
     "dict_learning_online"        => gDEC.dict_learning_online,
     "fastica"                     => gDEC.fastica,
     "non_negative_factorization"  => gDEC.non_negative_factorization,
     "sparse_encode"               => gDEC.sparse_encode,
     "GenericUnivariateSelect"     => gFS.GenericUnivariateSelect,
     "SelectPercentile"            => gFS.SelectPercentile,
     "SelectKBest"                 => gFS.SelectKBest,
     "SelectFpr"                   => gFS.SelectFpr,
     "SelectFdr"                   => gFS.SelectFdr,
     "SelectFromModel"             => gFS.SelectFromModel,
     "SelectFwe"                   => gFS.SelectFwe,
     "RFE"                         => gFS.RFE,
     "RFECV"                       => gFS.RFECV,
     "VarianceThreshold"           => gFS.VarianceThreshold,
     "chi2"                        => gFS.chi2,
     "f_classif"                   => gFS.f_classif,
     "f_regression"                => gFS.f_regression,
     "mutual_info_classif"         => gFS.mutual_info_classif,
     "mutual_info_regression"      => gFS.mutual_info_regression,
     "SimpleImputer"               => gIMP.SimpleImputer,
     #"IterativeImputer"           => gIMP.IterativeImputer,
     #"KNNImputer"                 => gIMP.KNNImputer,
     "MissingIndicator"            => gIMP.MissingIndicator,
     "Binarizer"                   => gPREP.Binarizer,
     "FunctionTransformer"         => gPREP.FunctionTransformer,
     "KBinsDiscretizer"            => gPREP.KBinsDiscretizer,
     "KernelCenterer"              => gPREP.KernelCenterer,
     "LabelBinarizer"              => gPREP.LabelBinarizer,
     "LabelEncoder"                => gPREP.LabelEncoder,
     "MultiLabelBinarizer"         => gPREP.MultiLabelBinarizer,
     "MaxAbsScaler"                => gPREP.MaxAbsScaler,
     "MinMaxScaler"                => gPREP.MinMaxScaler,
     "Normalizer"                  => gPREP.Normalizer,
     "OneHotEncoder"               => gPREP.OneHotEncoder,
     "OrdinalEncoder"              => gPREP.OrdinalEncoder,
     "PolynomialFeatures"          => gPREP.PolynomialFeatures,
     "PowerTransformer"            => gPREP.PowerTransformer,
     "QuantileTransformer"         => gPREP.QuantileTransformer,
     "RobustScaler"                => gPREP.RobustScaler,
     "StandardScaler"              => gPREP.StandardScaler
     #"add_dummy_feature"          => gPREP.add_dummy_feature,
     #"binarize"                   => gPREP.binarize,
     #"label_binarize"             => gPREP.label_binarize,
     #"maxabs_scale"               => gPREP.maxabs_scale,
     #"minmax_scale"               => gPREP.minmax_scale,
     #"normalize"                  => gPREP.normalize,
     #"quantile_transform"         => gPREP.quantile_transform,
     #"robust_scale"               => gPREP.robust_scale,
     #"scale"                      => gPREP.scale,
     #"power_transform"            => gPREP.power_transform
    )
end

"""
    SKPreprocessor(preprocessor::String,args::Dict=Dict())

A wrapper for Scikitlearn preprocessor functions. 
Invoking `skpreprocessors()` will list the acceptable 
and supported functions. Please check Scikitlearn
documentation for arguments to pass.

Implements `fit!` and `transform!`.
"""
mutable struct SKPreprocessor <: Transformer
  name::String
  model::Dict
  args::Dict

  function SKPreprocessor(args=Dict())
    default_args=Dict(
                      :name => "skprep",
                      :preprocessor => "PCA",
                      :impl_args => Dict(),
                      :autocomponent=>false
                     )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    prep = cargs[:preprocessor]
    if !(prep in keys(preprocessor_dict)) 
      println("$prep is not supported.") 
      println()
      skpreprocessors()
      error("Argument keyword error")
    end
    new(cargs[:name],Dict(),cargs)
  end
end

function SKPreprocessor(prep::String,args::Dict=Dict())
  SKPreprocessor(Dict(:preprocessor => prep,:name=>prep,args...))
end

function skpreprocessors()
  processors = keys(preprocessor_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: SKPreprocessor(name::String, args::Dict=Dict())")
  println("where *name* can be one of:")
  println()
  [print(processor," ") for processor in processors]
  println()
  println()
  println("and *args* are the corresponding preprocessor's initial parameters.")
  println("Note: Please consult Scikitlearn's online help for more details about the preprocessor's arguments.")
end

function fit!(skp::SKPreprocessor, x::DataFrame, y::Vector=[])
  features = x |> Array
  impl_args = copy(skp.args[:impl_args])
  autocomp = skp.args[:autocomponent]
  if autocomp == true
    cols = ncol(x)
    ncomponents = 1
    if cols > 0
      ncomponents = round(sqrt(cols),digits=0) |> Integer
      push!(impl_args,:n_components => ncomponents)
    end
  end
  preprocessor = skp.args[:preprocessor]
  py_preprocessor = preprocessor_dict[preprocessor]

  # Train model
  preproc = py_preprocessor(;impl_args...)
  preproc.fit(features)
  skp.model = Dict(
                   :skpreprocessor => preproc,
                   :impl_args => impl_args
                  )
end

function transform!(skp::SKPreprocessor, x::DataFrame)
  features = deepcopy(x) |> Array
  model=skp.model[:skpreprocessor]
  return collect(model.transform(features)) |> DataFrame
end

end

