module SKLearners

using PyCall

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export SKLearner, sklearners


function __init__()
  global gENS  = pyimport_conda("sklearn.ensemble","scikit-learn")
  global gLM   = pyimport_conda("sklearn.linear_model","scikit-learn")
  global gDA   = pyimport_conda("sklearn.discriminant_analysis","scikit-learn")
  global gNN   = pyimport_conda("sklearn.neighbors","scikit-learn")
  global gSVM  = pyimport_conda("sklearn.svm","scikit-learn")
  global gTREE = pyimport_conda("sklearn.tree","scikit-learn")
  global gANN  = pyimport_conda("sklearn.neural_network","scikit-learn")
  global gGP   = pyimport_conda("sklearn.gaussian_process","scikit-learn")
  global gKR   = pyimport_conda("sklearn.kernel_ridge","scikit-learn")
  global gNB   = pyimport_conda("sklearn.naive_bayes","scikit-learn")
  global gISO  = pyimport_conda("sklearn.isotonic","scikit-learn")

  # Available scikit-learn learners.
  global learner_dict = Dict(
       "AdaBoostClassifier"          => gENS.AdaBoostClassifier,
       "BaggingClassifier"           => gENS.BaggingClassifier,
       "ExtraTreesClassifier"        => gENS.ExtraTreesClassifier,
       "VotingClassifier"            => gENS.VotingClassifier,
       "GradientBoostingClassifier"  => gENS.GradientBoostingClassifier,
       "RandomForestClassifier"      => gENS.RandomForestClassifier,
       "LDA"                         => gDA.LinearDiscriminantAnalysis,
       "QDA"                         => gDA.QuadraticDiscriminantAnalysis,
       "LogisticRegression"          => gLM.LogisticRegression,
       "PassiveAggressiveClassifier" => gLM.PassiveAggressiveClassifier,
       "RidgeClassifier"             => gLM.RidgeClassifier,
       "RidgeClassifierCV"           => gLM.RidgeClassifierCV,
       "SGDClassifier"               => gLM.SGDClassifier,
       "KNeighborsClassifier"        => gNN.KNeighborsClassifier,
       "RadiusNeighborsClassifier"   => gNN.RadiusNeighborsClassifier,
       "NearestCentroid"             => gNN.NearestCentroid,
       "SVC"                         => gSVM.SVC,
       "LinearSVC"                   => gSVM.LinearSVC,
       "NuSVC"                       => gSVM.NuSVC,
       "MLPClassifier"               => gANN.MLPClassifier,
       "GaussianProcessClassifier"   => gGP.GaussianProcessClassifier,
       "DecisionTreeClassifier"      => gTREE.DecisionTreeClassifier,
       "GaussianNB"                  => gNB.GaussianNB,
       "MultinomialNB"               => gNB.MultinomialNB,
       "ComplementNB"                => gNB.ComplementNB,
       "BernoulliNB"                 => gNB.BernoulliNB,
       "SVR"                         => gSVM.SVR,
       "Ridge"                       => gLM.Ridge,
       "RidgeCV"                     => gLM.RidgeCV,
       "Lasso"                       => gLM.Lasso,
       "ElasticNet"                  => gLM.ElasticNet,
       "Lars"                        => gLM.Lars,
       "LassoLars"                   => gLM.LassoLars,
       "OrthogonalMatchingPursuit"   => gLM.OrthogonalMatchingPursuit,
       "BayesianRidge"               => gLM.BayesianRidge,
       "ARDRegression"               => gLM.ARDRegression,
       "SGDRegressor"                => gLM.SGDRegressor,
       "PassiveAggressiveRegressor"  => gLM.PassiveAggressiveRegressor,
       "KernelRidge"                 => gKR.KernelRidge,
       "KNeighborsRegressor"         => gNN.KNeighborsRegressor,
       "RadiusNeighborsRegressor"    => gNN.RadiusNeighborsRegressor,
       "GaussianProcessRegressor"    => gGP.GaussianProcessRegressor,
       "DecisionTreeRegressor"       => gTREE.DecisionTreeRegressor,
       "RandomForestRegressor"       => gENS.RandomForestRegressor,
       "ExtraTreesRegressor"         => gENS.ExtraTreesRegressor,
       "AdaBoostRegressor"           => gENS.AdaBoostRegressor,
       "GradientBoostingRegressor"   => gENS.GradientBoostingRegressor,
       "IsotonicRegression"          => gISO.IsotonicRegression,
       "MLPRegressor"                => gANN.MLPRegressor
      )
end

"""
    SKLearner(learner::String, args::Dict=Dict())

A Scikitlearn wrapper to load the different machine learning models.
Invoking `sklearners()` will list the available learners. Please
consult Scikitlearn documentation for arguments to pass.

Implements `fit!` and `transform!`. 
"""
mutable struct SKLearner <: Learner
  name::String
  model::Dict
  args::Dict

  function SKLearner(args=Dict())
    default_args=Dict(
       :name => "sklearner",
       :output => :class,
       :learner => "LinearSVC",
       :impl_args => Dict()
      )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    skl = cargs[:learner]
    if !(skl in keys(learner_dict)) 
      println("$skl is not supported.") 
      println()
      sklearners()
      error("Argument keyword error")
    end
    new(cargs[:name],Dict(),cargs)
  end
end

function SKLearner(learner::String, args::Dict=Dict())
  SKLearner(Dict(:learner => learner,:name=>learner,  args...))
end

"""
    sklearners()

List the available scikitlearn machine learners.
"""
function sklearners()
  learners = keys(learner_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: SKLearner(name::String, args::Dict=Dict())")
  println("where 'name' can be one of:")
  println()
  [print(learner," ") for learner in learners]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult Scikitlearn's online help for more details about the learner's arguments.")
end

function fit!(skl::SKLearner, xx::DataFrame, y::Vector)
  x = xx |> Array
  impl_args = copy(skl.args[:impl_args])
  learner = skl.args[:learner]
  py_learner = learner_dict[learner]

  # Assign CombineML-specific defaults if required
  if learner == "RadiusNeighborsClassifier"
    if get(impl_args, :outlier_label, nothing) == nothing
      impl_options[:outlier_label] = labels[rand(1:size(labels, 1))]
    end
  end

  # Train
  modelobj = py_learner(;impl_args...)
  modelobj.fit(x,y)
  skl.model = Dict(
      :sklearner => modelobj,
      :impl_args => impl_args
     )
end


function transform!(skl::SKLearner, xx::DataFrame)
	x = deepcopy(xx) |> Array
  #return collect(skl.model[:predict](x))
  sklearner = skl.model[:sklearner]
  return collect(sklearner.predict(x))
end

end

