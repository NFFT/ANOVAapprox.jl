var documenterSearchIndex = {"docs":
[{"location":"about.html#About","page":"About","title":"About","text":"","category":"section"},{"location":"about.html","page":"About","title":"About","text":"ANOVAapprox.jl has been developed during the research for several papers. It is currently maintained by Michael Schmischke (michael.schmischke@math.tu-chemnitz.de) with contributions from Felix Bartel.","category":"page"},{"location":"about.html","page":"About","title":"About","text":"If you want to contribute or have any questions, visit the GitHub repository to clone/fork the repository or open an issue.","category":"page"},{"location":"approx.html#Approximation","page":"Approximation","title":"Approximation","text":"","category":"section"},{"location":"approx.html","page":"Approximation","title":"Approximation","text":"    CurrentModule = ANOVAapprox","category":"page"},{"location":"approx.html","page":"Approximation","title":"Approximation","text":"Modules = [ANOVAapprox]\nPages   = [\"approx.jl\"]","category":"page"},{"location":"approx.html#ANOVAapprox.approx","page":"Approximation","title":"ANOVAapprox.approx","text":"approx\n\nA struct to hold the scattered data function approximation.\n\nFields\n\nbasis::String - basis of the function space; currently choice of \"per\", \"cos\", \"cheb\",\"std\", \"wav1\", \"wav2\",\"wav3\",\"wav4\"\nX::Matrix{Float64} - scattered data nodes with d rows and M columns\ny::Union{Vector{ComplexF64},Vector{Float64}} - M function values (complex for basis = \"per\", real ortherwise)\nU::Vector{Vector{Int}} - a vector containing susbets of coordinate indices\nN::Vector{Int} - bandwdiths for each ANOVA term\ntrafo::GroupedTransform - holds the grouped transformation\nfc::Dict{Float64,GroupedCoefficients} - holds the GroupedCoefficients after approximation for every different regularization parameters\n\nConstructor\n\napprox( X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, U::Vector{Vector{Int}}, N::Vector{Int}, basis::String = \"cos\" )\n\nAdditional Constructor\n\napprox( X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, ds::Int, N::Vector{Int}, basis::String = \"cos\" )\n\n\n\n\n\n","category":"type"},{"location":"approx.html#ANOVAapprox.approximate-Tuple{ANOVAapprox.approx, Float64}","page":"Approximation","title":"ANOVAapprox.approximate","text":"approximate( a::approx, λ::Float64; max_iter::Int = 50, weights::Union{Vector{Float64},Nothing} = nothing, verbose::Bool = false, solver::String = \"lsqr\" )::Nothing\n\nThis function computes the approximation for the regularization parameter lambda.\n\n\n\n\n\n","category":"method"},{"location":"approx.html#ANOVAapprox.approximate-Tuple{ANOVAapprox.approx}","page":"Approximation","title":"ANOVAapprox.approximate","text":"approximate( a::approx; lambda::Vector{Float64} = exp.(range(0, 5, length = 5)), max_iter::Int = 50, weights::Union{Vector{Float64},Nothing} = nothing, verbose::Bool = false, solver::String = \"lsqr\" )::Nothing\n\nThis function computes the approximation for the regularization parameters contained in lambda.\n\n\n\n\n\n","category":"method"},{"location":"approx.html#ANOVAapprox.evaluate-Tuple{ANOVAapprox.approx, Float64}","page":"Approximation","title":"ANOVAapprox.evaluate","text":"evaluate( a::approx; λ::Float64 )::Union{Vector{ComplexF64},Vector{Float64}}\n\nThis function evaluates the approximation on the nodes a.X for the regularization parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"approx.html#ANOVAapprox.evaluate-Tuple{ANOVAapprox.approx, Matrix{Float64}, Float64}","page":"Approximation","title":"ANOVAapprox.evaluate","text":"evaluate( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Vector{ComplexF64},Vector{Float64}}\n\nThis function evaluates the approximation on the nodes X for the regularization parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"approx.html#ANOVAapprox.evaluate-Tuple{ANOVAapprox.approx, Matrix{Float64}}","page":"Approximation","title":"ANOVAapprox.evaluate","text":"evaluate( a::approx; X::Matrix{Float64} )::Union{Vector{ComplexF64},Vector{Float64}}\n\nThis function evaluates the approximation on the nodes X for all regularization parameters.\n\n\n\n\n\n","category":"method"},{"location":"approx.html#ANOVAapprox.evaluate-Tuple{ANOVAapprox.approx}","page":"Approximation","title":"ANOVAapprox.evaluate","text":"evaluate( a::approx )::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}\n\nThis function evaluates the approximation on the nodes a.X for all regularization parameters.\n\n\n\n\n\n","category":"method"},{"location":"analysis.html#Analysis","page":"Analysis","title":"Analysis","text":"","category":"section"},{"location":"analysis.html","page":"Analysis","title":"Analysis","text":"    CurrentModule = ANOVAapprox","category":"page"},{"location":"analysis.html","page":"Analysis","title":"Analysis","text":"Modules = [ANOVAapprox]\nPages   = [\"analysis.jl\"]","category":"page"},{"location":"analysis.html#ANOVAapprox.get_AttributeRanking-Tuple{ANOVAapprox.approx, Float64}","page":"Analysis","title":"ANOVAapprox.get_AttributeRanking","text":"get_AttributeRanking( a::approx, λ::Float64 )::Vector{Float64}\n\nThis function returns the attribute ranking of the approximation for reg. parameter lambda as a vector of length a.d.\n\n\n\n\n\n","category":"method"},{"location":"analysis.html#ANOVAapprox.get_AttributeRanking-Tuple{ANOVAapprox.approx}","page":"Analysis","title":"ANOVAapprox.get_AttributeRanking","text":"get_AttributeRanking( a::approx, λ::Float64 )::Dict{Float64,Vector{Float64}}\n\nThis function returns the attribute ranking of the approximation for all reg. parameters lambda as a dictionary of vectors of length a.d.\n\n\n\n\n\n","category":"method"},{"location":"analysis.html#ANOVAapprox.get_GSI-Tuple{ANOVAapprox.approx, Float64}","page":"Analysis","title":"ANOVAapprox.get_GSI","text":"get_GSI( a::approx, λ::Float64; dict::Bool = false )::Union{Vector{Float64},Dict{Vector{Int},Float64}}\n\nThis function returns the global sensitivity indices of the approximation with lambda as a vector for dict = false or else a dictionary.\n\n\n\n\n\n","category":"method"},{"location":"analysis.html#ANOVAapprox.get_GSI-Tuple{ANOVAapprox.approx}","page":"Analysis","title":"ANOVAapprox.get_GSI","text":"get_GSI( a::approx; dict::Bool = false )::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}\n\nThis function returns the global sensitivity indices of the approximation for all lambda as a vector for dict = false or else a dictionary.\n\n\n\n\n\n","category":"method"},{"location":"analysis.html#ANOVAapprox.get_ShapleyValues-Tuple{ANOVAapprox.approx, Float64}","page":"Analysis","title":"ANOVAapprox.get_ShapleyValues","text":"get_ShapleyValues( a::approx, λ::Float64 )::Vector{Float64}\n\nThis function returns the Shapley values of the approximation for reg. parameter lambda as a vector of length a.d.\n\n\n\n\n\n","category":"method"},{"location":"analysis.html#ANOVAapprox.get_ShapleyValues-Tuple{ANOVAapprox.approx}","page":"Analysis","title":"ANOVAapprox.get_ShapleyValues","text":"This function returns the Shapley values of the approximation for all reg. parameters lambda as a dictionary of vectors of length a.d.\n\n\n\n\n\n","category":"method"},{"location":"analysis.html#ANOVAapprox.get_variances-Tuple{ANOVAapprox.approx, Float64}","page":"Analysis","title":"ANOVAapprox.get_variances","text":"get_variances( a::approx, λ::Float64; dict::Bool = false )::Union{Vector{Float64},Dict{Vector{Int},Float64}}\n\nThis function returns the variances of the approximated ANOVA terms with lambda as a vector for dict = false or else a dictionary.\n\n\n\n\n\n","category":"method"},{"location":"analysis.html#ANOVAapprox.get_variances-Tuple{ANOVAapprox.approx}","page":"Analysis","title":"ANOVAapprox.get_variances","text":"get_variances( a::approx; dict::Bool = false )::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}\n\nThis function returns the variances of the approximated ANOVA terms for all lambda as a vector for dict = false or else a dictionary.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#Errors","page":"Errors","title":"Errors","text":"","category":"section"},{"location":"errors.html","page":"Errors","title":"Errors","text":"    CurrentModule = ANOVAapprox","category":"page"},{"location":"errors.html","page":"Errors","title":"Errors","text":"Modules = [ANOVAapprox]\nPages   = [\"errors.jl\"]","category":"page"},{"location":"errors.html#ANOVAapprox.get_L2error-Tuple{ANOVAapprox.approx, Float64, Function, Float64}","page":"Errors","title":"ANOVAapprox.get_L2error","text":"get_L2error( a::approx, norm::Float64, bc_fun::Function, λ::Float64 )::Float64\n\nThis function computes the relative L_2 error of the function given the norm norm and a function that returns the basis coefficients bc_fun for regularization parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_L2error-Tuple{ANOVAapprox.approx, Float64, Function}","page":"Errors","title":"ANOVAapprox.get_L2error","text":"get_L2error( a::approx, norm::Float64, bc_fun::Function )::Dict{Float64,Float64}\n\nThis function computes the relative L_2 error of the function given the norm norm and a function that returns the basis coefficients bc_fun for all regularization parameters.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_l2error-Tuple{ANOVAapprox.approx, Float64}","page":"Errors","title":"ANOVAapprox.get_l2error","text":"get_l2error( a::approx, λ::Float64 )::Float64\n\nThis function computes the relative ell_2 error on the training nodes for regularization parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_l2error-Tuple{ANOVAapprox.approx, Matrix{Float64}, Union{Vector{ComplexF64}, Vector{Float64}}, Float64}","page":"Errors","title":"ANOVAapprox.get_l2error","text":"get_l2error( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, λ::Float64 )::Float64\n\nThis function computes the relative ell_2 error on the data X and y for regularization parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_l2error-Tuple{ANOVAapprox.approx, Matrix{Float64}, Union{Vector{ComplexF64}, Vector{Float64}}}","page":"Errors","title":"ANOVAapprox.get_l2error","text":"get_l2error( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, )::Dict{Float64,Float64}\n\nThis function computes the relative ell_2 error on the data X and y for all regularization parameters.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_l2error-Tuple{ANOVAapprox.approx}","page":"Errors","title":"ANOVAapprox.get_l2error","text":"get_l2error( a::approx )::Dict{Float64,Float64}\n\nThis function computes the relative ell_2 error on the training nodes for all regularization parameters.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_mad-Tuple{ANOVAapprox.approx, Float64}","page":"Errors","title":"ANOVAapprox.get_mad","text":"get_mad( a::approx, λ::Float64 )::Float64\n\nThis function computes the mean absolute deviation (mad) on the training nodes for regularization parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_mad-Tuple{ANOVAapprox.approx, Matrix{Float64}, Union{Vector{ComplexF64}, Vector{Float64}}, Float64}","page":"Errors","title":"ANOVAapprox.get_mad","text":"get_mad( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, λ::Float64 )::Float64\n\nThis function computes the mean absolute deviation (mad) on the data X and y for regularization parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_mad-Tuple{ANOVAapprox.approx, Matrix{Float64}, Union{Vector{ComplexF64}, Vector{Float64}}}","page":"Errors","title":"ANOVAapprox.get_mad","text":"get_mse( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, )::Dict{Float64,Float64}\n\nThis function computes the mean absolute deviation (mad) on the data X and y for all regularization parameters.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_mad-Tuple{ANOVAapprox.approx}","page":"Errors","title":"ANOVAapprox.get_mad","text":"get_mad( a::approx )::Dict{Float64,Float64}\n\nThis function computes the mean absolute deviation (mad) on the training nodes for all regularization parameters.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_mse-Tuple{ANOVAapprox.approx, Float64}","page":"Errors","title":"ANOVAapprox.get_mse","text":"get_mse( a::approx, λ::Float64 )::Float64\n\nThis function computes the mean square error (mse) on the training nodes for regularization parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_mse-Tuple{ANOVAapprox.approx, Matrix{Float64}, Union{Vector{ComplexF64}, Vector{Float64}}, Float64}","page":"Errors","title":"ANOVAapprox.get_mse","text":"get_mse( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, λ::Float64 )::Float64\n\nThis function computes the mean square error (mse) on the data X and y for regularization parameter λ.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_mse-Tuple{ANOVAapprox.approx, Matrix{Float64}, Union{Vector{ComplexF64}, Vector{Float64}}}","page":"Errors","title":"ANOVAapprox.get_mse","text":"get_mse( a::approx, X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, )::Dict{Float64,Float64}\n\nThis function computes the mean square error (mse) on the data X and y for all regularization parameters.\n\n\n\n\n\n","category":"method"},{"location":"errors.html#ANOVAapprox.get_mse-Tuple{ANOVAapprox.approx}","page":"Errors","title":"ANOVAapprox.get_mse","text":"get_mse( a::approx )::Dict{Float64,Float64}\n\nThis function computes the mean square error (mse) on the training nodes for all regularization parameters.\n\n\n\n\n\n","category":"method"},{"location":"index.html#Welcome-to-ANOVAapprox.jl","page":"Home","title":"Welcome to ANOVAapprox.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"This package provides a framework for the method ANOVAapprox to approximate high-dimensional functions with a low superposition dimension or a sparse ANOVA decomposition from scattered data. ","category":"page"},{"location":"index.html#Literature","page":"Home","title":"Literature","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"<ul>\n  <li>D. Potts und M. Schmischke <br> \n  <b>Interpretable transformed ANOVA approximation on the example of the prevention of forest fires</b> <br>\n  <a href=\"https://arxiv.org/abs/2110.07353\">arXiv</a>, <a href=\"https://www-user.tu-chemnitz.de/~mischmi/papers/transformedanova.pdf\">PDF</a></li>\n  <li>F. Bartel, D. Potts und M. Schmischke <br> \n  <b>Grouped transformations and Regularization in high-dimensional explainable ANOVA approximation</b> <br>\n  SIAM Journal on Scientific Computing (accepted) <br>\n  <a href=\"https://arxiv.org/abs/2010.10199\">arXiv</a>, <a href=\"https://www-user.tu-chemnitz.de/~mischmi/papers/groupedtransforms.pdf\">PDF</a></li>\n  <li>D. Potts und M. Schmischke <br> \n  <b>Interpretable approximation of high-dimensional data</b> <br>\n  SIAM Journal on Mathematics of Data Science (accepted) <br>\n  <a href=\"https://arxiv.org/abs/2103.13787\">arXiv</a>, <a href=\"https://www-user.tu-chemnitz.de/~mischmi/papers/attributeranking.pdf\">PDF</a>, <a href=\"https://github.com/NFFT/AttributeRankingExamples\">Software</a></li>\n  <li>D. Potts und M. Schmischke <br> \n  <b>Learning multivariate functions with low-dimensional structures using polynomial bases</b><br>\n  Journal of Computational and Applied Mathematics 403, 113821, 2021<br>\n  <a href=\"https://doi.org/10.1016/j.cam.2021.113821\">DOI</a>, <a href=\"https://arxiv.org/abs/1912.03195\">arXiv</a>, <a href=\"https://www-user.tu-chemnitz.de/~mischmi/papers/anovacube.pdf\">PDF</a></li>\n  <li>D. Potts und M. Schmischke <br> \n  <b>Approximation of high-dimensional periodic functions with Fourier-based methods</b><br>\n  SIAM Journal on Numerical Analysis 59 (5), 2393-2429, 2021<br>\n  <a href=\"https://doi.org/10.1137/20M1354921\">DOI</a>, <a href=\"https://arxiv.org/abs/1907.11412\">arXiv</a>, <a href=\"https://www-user.tu-chemnitz.de/~mischmi/papers/anovafourier.pdf\">PDF</a></li>\n</ul>","category":"page"}]
}
