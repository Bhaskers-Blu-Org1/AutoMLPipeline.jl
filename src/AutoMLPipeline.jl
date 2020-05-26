module AutoMLPipeline

greet() = print("Hello World!")
export fit!, transform!, fit_transform!

include("abstracttypes.jl")
using .AbsTypes
export Machine, Computer, Workflow, Learner, Transformer

include("utils.jl")
using .Utils
export getiris, getprofb

include("baselines.jl")
using .Baselines
export Baseline, Identity

include("basefilters.jl")
using .BaseFilters
export OneHotEncoder, Imputer


include("decisiontree.jl")
using .DecisionTreeLearners
export PrunedTree, RandomForest, Adaboost

include("ensemble.jl")
using .EnsembleMethods
export VoteEnsemble, StackEnsemble, BestLearner

include("pipelines.jl")
using .Pipelines
export @pipeline, @pipelinex, @pipelinez
export Pipeline, ComboPipeline 

include("crossvalidator.jl")
using .CrossValidators
export crossvalidate

include("skcrossvalidator.jl")
using .SKCrossValidators
export crossvalidate

include("skpreprocessor.jl")
using .SKPreprocessors
export SKPreprocessor, skpreprocessors

include("sklearners.jl")
using .SKLearners
export SKLearner, sklearners

include("naremover.jl")
using .NARemovers
export NARemover

include("jlpreprocessors.jl")
using .JLPreprocessors
export JLPreprocessor
export testjlprep

include("xgbc.jl")
using .XGBoostLearners
export Xgbc

include("featureselector.jl")
using .FeatureSelectors
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator

end # module
