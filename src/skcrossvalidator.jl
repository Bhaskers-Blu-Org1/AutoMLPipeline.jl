module SKCrossValidators

import PyCall

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.CrossValidators.crossvalidate
export crossvalidate

function __init__()
    global gSKM = PyCall.pyimport_conda("sklearn.metrics","scikit-learn")
    global metric_dict = Dict(
          "roc_auc_score"                   => gSKM.roc_auc_score,
          "accuracy_score"                  => gSKM.accuracy_score,
          "auc"                             => gSKM.auc,
          "average_precision_score"         => gSKM.average_precision_score,
          "balanced_accuracy_score"         => gSKM.balanced_accuracy_score,
          "brier_score_loss"                => gSKM.brier_score_loss,
          "classification_report"           => gSKM.classification_report,
          "cohen_kappa_score"               => gSKM.cohen_kappa_score,
          "confusion_matrix"                => gSKM.confusion_matrix,
          "f1_score"                        => gSKM.f1_score,
          "fbeta_score"                     => gSKM.fbeta_score,
          "hamming_loss"                    => gSKM.hamming_loss,
          "hinge_loss"                      => gSKM.hinge_loss,
          "log_loss"                        => gSKM.log_loss,
          "matthews_corrcoef"               => gSKM.matthews_corrcoef,
          "multilabel_confusion_matrix"     => gSKM.multilabel_confusion_matrix,
          "precision_recall_curve"          => gSKM.precision_recall_curve,
          "precision_recall_fscore_support" => gSKM.precision_recall_fscore_support,
          "precision_score"                 => gSKM.precision_score,
          "recall_score"                    => gSKM.recall_score,
          "roc_auc_score"                   => gSKM.roc_auc_score,
          "roc_curve"                       => gSKM.roc_curve,
          "jaccard_score"                   => gSKM.jaccard_score,
          "zero_one_loss"                   => gSKM.zero_one_loss
         )
end

function checkfun(sfunc::String)
    if !(sfunc in keys(metric_dict))
        println("$sfunc metric is not supported")
        println("metric: ",keys(metric_dict))
        error("Metric keyword error")
    end
end

"""
    crossvalidate(pl::Machine,X::DataFrame,Y::Vector,sfunc::String="balanced_accuracy_score",nfolds=10)

Runs K-fold cross-validation using balanced accuracy as the default. It support the 
following metric:
- accuracy_score
- balanced_accuracy_score
- cohen_kappa_score
- jaccard_score
- matthews_corrcoef
- hamming_loss
- zero_one_loss
- f1_score
- precision_score
- recall_score
"""
function crossvalidate(pl::Machine,X::DataFrame,Y::Vector,
                       sfunc::String="accuracy_score",nfolds=10,verbose::Bool=true)
    checkfun(sfunc)
    pfunc = metric_dict[sfunc]
    metric(a,b) = pfunc(a,b)
    crossvalidate(pl,X,Y,metric,nfolds,verbose)
end

function crossvalidate(pl::Machine,X::DataFrame,Y::Vector,
                       sfunc::String,averagetype::String,nfolds=10,verbose::Bool=true)
    checkfun(sfunc)
    pfunc = metric_dict[sfunc]
    metric(a,b) = pfunc(a,b,average=averagetype)
    crossvalidate(pl,X,Y,metric,nfolds,verbose)
end


end
