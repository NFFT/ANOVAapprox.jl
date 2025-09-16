#import Pkg; Pkg.add("ANOVAapprox")

# Example for approximating an periodic function

using ANOVAapprox 
using Random
using Plots
using Plots.PlotMeasures
gaston()

function TestFunction(x::Vector{Float64})::Float64
    return x[1]*x[5] + 2 - exp(x[4]) + sqrt(x[6]+3+x[2])
end

rng = MersenneTwister(1234)

##################################
## Definition of the parameters ##
##################################

d = 6 # dimension

M        =  10_000 # number of used evaluation points to train the model 
M_test   = 100_000 # number of used evaluation points to test the accuracity the model 

max_iter = 50 # maximum number of iterations

# there are 3 possibilities with varying degree of freedom to define the number of used frequencies 
########### Variant 1:
ds  = 2         # superposition dimension
num = sum(binomial.(d,1:ds)) # number of used subsets
b   =  M / (log10(M) * num)  # number for the number of frequencies if we use logarithmic oversampling and distribute it evenly to all subsets
bw  = [floor(Int,b/2)*2, floor(Int,sqrt(b)/2)*2] # bandwidths (use even numbers)
# Use all subsets up to ds and use bw[1] many frequences in the the subsets with one element, b[2]^2 many for subsets with two elements and so on 
#
########### Variant 2:
# used subsets:
# U = [Int[], [1], [2], [3], [4], [5], [6],
#     [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]
# Bandwidths for these subsets:
# N = [[0]  , 100, 100, 100, 100, 100, 100,
#         10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10]
# Use the subsets U with the bandwiths N. The bandwith N[i] corresponds to the subset U[i]. For subsets with more then one direction is the same bandwidth in all directions used 
#
########### Variant 3:
# used subsets:
# U = [Int[],   [1],   [2],   [3],   [4],   [5],   [6],
#      [ 1, 2],[ 1, 3],[ 1, 4],[ 1, 5],[ 1, 6],[ 2, 3],[ 2, 4],[ 2, 5],[ 2, 6],[ 3, 4],[ 3, 5],[ 3, 6],[ 4, 5],[ 4, 6],[ 5, 6]]
# Bandwidths for these subsets:
# N = [Int[], [100], [100], [100], [100], [100], [100],
#      [10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10]]
# Use the subsets U with the bandwiths N. The bandwith N[i] corresponds to the subset U[i]. The bandwidth N[i][j] corresponds to the direction U[i][j]

λs = [0.0, 1.0] # used regularisation parameters λ 

############################
## Generation of the data ##
############################

X      = 2 .* rand(rng, d, M) .- 1           # get random points
X      = sin.(pi .* (X .- 0.5))                 # distribute the points along the inverse distribution
y      = [TestFunction(X[:, i]) for i = 1:M] # evaluate the function at these points
X_test      = 2 .* rand(rng, d, M_test) .- 1                #
X_test      = sin.(pi .* (X_test .- 0.5))                      #
y_test = [TestFunction(X_test[:, i]) for i = 1:M_test] # the same for the test points

##########################
## Do the approximation ##
##########################

ads = ANOVAapprox.approx(X, y, ds, bw, "cheb") # generate the data structure for the approximation
ANOVAapprox.approximate(ads, lambda = λs)   # do the approximation for all specified regularisation parameters

################################
## get approximation accuracy ##
################################

# mse = ANOVAapprox.get_mse(ads) # get mse error at the given training points
mse = ANOVAapprox.get_mse(ads, X_test, y_test) # get mse error at the test points
mse_min, λ_min = findmin(mse) # get the regularisation parameter which leads to the minimal error
println("mse = " * string(mse_min))

###############################################
## Analyze the model to improve the accuracy ##
###############################################


ar = ANOVAapprox.get_AttributeRanking(ads, λ_min) # get the attrbute ranking
p1 = Plots.plot(1:d, ar, markershape=:utriangle, st=:sticks, legend = false, yaxis = :log10, ylims=(10^(minimum(log10.(ar))-0.5), 1), title = "Attribute Ranking") # plot the arrtibute ranking in an logplot
println("active dimensions: "*string(collect(1:d)[ar.>1e-2]))

gsis  = ANOVAapprox.get_GSI(ads, λ_min) # get the gsis
label = string.(ads.U[2:end])
l     = length(label)
p2    = Plots.plot(gsis, xticks = (1:l, label), markershape=:utriangle, st=:sticks, legend = false, yaxis = :log10, ylims=(10^(minimum(log10.(gsis))-0.5), 1), reuse = false, title = "Global sensitivity indices") # plot the gsis
println("important dimensional interactions: "*string(label[gsis.>1e-10]))

################################################
## Approximation with better suited index set ##
################################################

U    = ads.U[append!([true],gsis.>1e-10)] # get important subsets
bws  = M / (log10(M) * (length(U) - 1))  # calculate frequencies per subset
N    = [floor(bws^(1/n)/2)*2 for n = length.(U)]    # distribute the frequencies evenly and make them even
N[1] = 0
N    = Int.(N)

a = ANOVAapprox.approx(X, y, U, N, "cheb") # generate the data structure for the approximation
ANOVAapprox.approximate(a, lambda = λs)   # do the approximation for all specified regularisation parameters

mse = ANOVAapprox.get_mse(a, X_test, y_test) # get mse error at the test points
mse_min, λ_min = findmin(mse) # get the regularisation parameter which leads to the minimal error
println("mse = " * string(mse_min))

########################
## Evaluate the model ##
########################

# y_approx = ANOVAapprox.evaluate(a, λ_min) # evaluate the approximation at the training points for the regularisation λ_min
# y_approx = ANOVAapprox.evaluate(a, X_test, λ_min) # evaluate the approximation at the points X_test for the regularisation λ_min

# In the following we plot the real and the approximated anova term for the subset u=[4]

y_eval_anova = ANOVAapprox.evaluateANOVAterms(a, X_test, λ_min) # evaluate all of the ANOVA terms 
pos = findfirst(==([4]), a.U) # find the index for the subset u=[4]
y_eval_anova_4 = y_eval_anova[:,pos]

perm = sortperm(X_test[4,:])
X_plot = X_test[4,perm]
y_eval_anova_4_plot = real.(y_eval_anova_4[perm])
y_anova_4_plot = -exp.(X_plot) .+ 1.2660446775548282

p3 = Plots.plot(X_plot,[y_eval_anova_4_plot,y_anova_4_plot], reuse = false, title = "Approximation of the ANOVA term 4", labels = ["approximation" "ANOVA term"])

display(p1); display(p2); p3
