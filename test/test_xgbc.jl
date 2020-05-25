module TestXgbc

using Test
using Random
using AutoMLPipeline
using AutoMLPipeline.GradientBoostingClassifiers
using AutoMLPipeline.CrossValidators
using AutoMLPipeline.Utils

function test_xgbc()
  Random.seed!(123)
  acc(X,Y) = score(:accuracy,X,Y)
  data=getiris()
  X=data[:,1:4] 
  Y=data[:,5] |> Vector{String}
  xgb = Xgbc(Dict(:silent=>true))
  fit!(xgb,X,Y)
  res=transform!(xgb,X)
  @test crossvalidate(xgb,X,Y,acc,10,false).mean > 90.0
end
@testset "Xgbc" begin
  test_xgbc()
end


end
