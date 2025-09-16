#import Pkg; Pkg.add("ANOVAapprox")

# Example for approximating an periodic function

using ANOVAapprox 
using Random
using Plots
using Plots.PlotMeasures
gaston()

function TestFunction(x::Vector{Float64})::Float64
    e = mod(abs(x[1]+im*x[2])+(angle(x[1]+im*x[2])/(pi*8)),0.25)>0.125
    return e*2-1
end

rng = MersenneTwister(1234)

##################################
## Definition of the parameters ##
##################################

d = 3 # dimension   

M        =  10_000 # number of used evaluation points to train the model 
M_test   =  10_000 # number of used evaluation points to test the accuracity the model 

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

λs = [0.0] # used regularisation parameters λ 

############################
## Generation of the data ##
############################

X      = rand(rng, d, M) .- 0.5              # construct the evaluation points for training
y      = [TestFunction(X[:, i]) for i = 1:M] # evaluate the function at these points
X      = X .+ 0.5
X_test = rand(rng, d, M_test) .- 0.5                   #
y_test = [TestFunction(X_test[:, i]) for i = 1:M_test] # the same for the test points
X_test = X_test .+ 0.5

###########################
## Do the classification ##
###########################

ads = ANOVAapprox.approx(X, y, ds, bw, "cos"; classification=true) # generate the data structure for the classification
ANOVAapprox.approximate(ads, lambda = λs)   # do the classification for all specified regularisation parameters

# #################################
# ## get classification accuracy ##
# #################################

y_approx = ANOVAapprox.evaluate(ads, X_test, 0.0) # evaluate the classification
acc = sum(sign.(y_approx).==y_test)/M_test # calculate the accuracity
println("accuracity = " * string(acc))

# ###############################################
# ## Analyze the model to improve the accuracy ##
# ###############################################


ar = ANOVAapprox.get_AttributeRanking(ads, 0.0) # get the attrbute ranking
p1 = Plots.plot(1:d, ar, markershape=:utriangle, st=:sticks, legend = false, yaxis = :log10, ylims=(10^(minimum(log10.(ar))-0.5), 1), title = "Attribute Ranking") # plot the arrtibute ranking in an logplot
println("active dimensions: "*string(collect(1:d)[ar.>1e-1]))

gsis  = ANOVAapprox.get_GSI(ads, 0.0) # get the gsis
label = string.(ads.U[2:end])
l     = length(label)
p2    = Plots.plot(gsis, xticks = (1:l, label), markershape=:utriangle, st=:sticks, legend = false, yaxis = :log10, ylims=(10^(minimum(log10.(gsis))-0.5), 1), reuse = false, title = "Global sensitivity indices") # plot the gsis
println("important dimensional interactions: "*string(label[gsis.>8*1e-2]))

#################################################
## Classification with better suited index set ##
#################################################

U    = ads.U[append!([true],gsis.>8*1e-2)] # get important subsets
bws  = M / (log10(M) * (length(U) - 1))  # calculate frequencies per subset
N    = [floor(bws^(1/n)/2)*2 for n = length.(U)]    # distribute the frequencies evenly and make them even
N[1] = 0
N    = Int.(N)

a = ANOVAapprox.approx(X, y, U, N, "cos") # generate the data structure for the classification
ANOVAapprox.approximate(a, lambda = λs)   # do the approximclassificationation for all specified regularisation parameters

y_approx = ANOVAapprox.evaluate(a, X_test, 0.0) # evaluate the classification
acc = sum(sign.(y_approx).==y_test)/M_test # calculate the accuracity
println("accuracity = " * string(acc))

########################
## Evaluate the model ##
########################

# y_approx = ANOVAapprox.evaluate(a, λ_min) # evaluate the classification at the training points for the regularisation λ_min
y_approx = ANOVAapprox.evaluate(a, X_test, 0.0) # evaluate the classification at the points X_test for the regularisation λ_min

p3 = Plots.scatter(X_test[1,:],X_test[2,:],zcolor=sign.(y_approx),color=:winter)

display(p1); display(p2); p3
