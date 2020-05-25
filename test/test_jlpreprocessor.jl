module TestJLPreprocessor
using Random
using DataFrames
using Test
using AutoMLPipeline
using AutoMLPipeline.JLPreprocessors
using AutoMLPipeline.Utils

function jltest()
  Random.seed!(123)
  iris=getiris()
  X = iris[:,1:(end-1)]
  Y = iris[:,end] |> collect
  for jprocs in ["ICA","PCA","PPCA","FA"]
    prep = JLPreprocessor(jprocs,Dict(:autocomponent=>true))
    iris = getiris()
    res=fit_transform!(prep,X)
	 @test (size(res) .== (nrow(X),2)) |> sum == 2
  end
end
@testset "JL PreProcessors" begin
  jltest()
end



end
