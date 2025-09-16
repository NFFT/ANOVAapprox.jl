using ANOVAapprox
using Plots
using StatsBase
gaston()

##################################
## Definition of the parameters ##
##################################

d = 8      # dimension
q = 2      # superposition dimension
M = 10_000 # number of samples
N = [5,2]  # number of parameters, should be vector of length q: 
# for wavelets the total number of parameters scales exponentially, i.e.:
# for q = 1 and N = [N1] the total number of parameters scales like ~O(d*2^N1)
# for q = 2 and N = [N1 , N2] the total number of parameters scales like ~O(d*2^N1) + O(d^2 * N2*2^N2)

λs = [0.0]      # regularization paramter
basis = "chui3" # choice of the basis functions

# for 'cos' the samples have to be in [0,1]^d
# for a periodic function use 'per' or wavelets  'chui2', 'chui3', 'chui4' (samples have to be in [-0.5,0.5]^d here, 'chuim' are the Chui-Wang wavelets of order m)

############################
## Generation of the data ##
############################

### define function 
function fun(X) 
    return 2 .* abs.(X[1,:]) + abs.(sin.(pi.* (X[2,:]) .* (X[3,:]))) + cos.(3 .* X[4,:])    # this function is of the form f_0 + f_1 + f_2 + f_3 + f_4 + f_2,3
end

### random points ####
if basis =="chui2" || basis =="chui3" || basis =="chui4" || basis =="per"
    X = rand( d, M) .-0.5        # for perioidic approximation samples have to be in [-0.5,0.5]^d
elseif basis == "cos"
    X = rand( d, M)
end
y = fun(X)

if basis == "chui1" || basis =="chui2" || basis =="chui3" || basis =="chui4" || basis =="per"
    X_test = rand( d, M) .- 0.5      # for perioidic approximation samples have to be in [-0.5,0.5]^d
elseif basis == "cos"
    X_test = rand( d, M)
end       
y_test = fun(X_test)

##########################
## Do the approximation ##
##########################

####  construct model for ANOVAapprox ####
anova_model = ANOVAapprox.approx(X, y, q, N, basis)

####  Do approximation by least-squares ###
ANOVAapprox.approximate(anova_model, lambda = λs)
println("Total number of used parameters = ", length(vec(anova_model.fc[λs[1]])))

#######################
## Analyze the model ##
#######################

### Do sensitivity analysis ####
gsis = ANOVAapprox.get_GSI(anova_model,0.0)                     #calculates indices for importance of terms (gsis is vector, with indices belonging to terms in anova_model.U)
gsis_as_dict = ANOVAapprox.get_GSI(anova_model,0.0,dict=true) 

#### plot gsis: (on the x-axis are the different ANOVA-terms, i.e. first d one-dimensional terms, then d*(d-1)/2 two-dimensional terms,...)
display(scatter(gsis,label="gsis", yscale=:log10, title="gsis", xlabel = "terms u" ))
png("gsis")                                                     # for the testfunction from above you can detect the terms u\in [1], [2], [3], [4], [2,3]

################################
## get approximation accuracy ##
################################

### error analysis ###
mse_train = ANOVAapprox.get_mse(anova_model,0.0)
mse_test = ANOVAapprox.get_mse(anova_model,X_test,y_test, 0.0)

println("MSE on test points:", mse_test) 

################################################
## Approximation with better suited index set ##
################################################

### Do the approximation again with adapted index-set U ####
U = ANOVAapprox.get_ActiveSet(anova_model, [0.01,0.01], 0.0)
println("Found index-set U: ",U )
anova_model = ANOVAapprox.approx(X, y, U, N .+ 2 , basis)               # increase number of paramers in N for the important terms
ANOVAapprox.approximate(anova_model, lambda = λs)
println("Total number of used parameters after ANOVA truncation: ", length(vec(anova_model.fc[λs[1]])))
mse_train = ANOVAapprox.get_mse(anova_model,0.0)
mse_test = ANOVAapprox.get_mse(anova_model,X_test,y_test, 0.0)
println("MSE on test points after ANOVA truncation:", mse_test) 