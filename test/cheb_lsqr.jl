#### NONPERIODIC TEST SOLVER LSQR ####

d = 8
ds = 2
M = 10_000
max_iter = 50
bw = [20, 4]
λs = [0.0, 1.0]

(X, y) = TestFunctionCheb.generateData(M)
(X_test, y_test) = TestFunctionCheb.generateData(M)

ads = ANOVAapprox.approx_ds(X, y, ds, bw, "cheb")
ANOVAapprox.approximate(ads, lambda = λs)

bw = ANOVAapprox.get_orderDependentBW(TestFunctionCheb.AS, bw)

aU = ANOVAapprox.approx_U(X, y, TestFunctionCheb.AS, bw, "cheb")
ANOVAapprox.approximate(aU, lambda = λs)

err_L2_ds = ANOVAapprox.get_L2error(ads, TestFunctionCheb.norm(), TestFunctionCheb.fc)[0.0]
err_L2_U = ANOVAapprox.get_L2error(aU, TestFunctionCheb.norm(), TestFunctionCheb.fc)[0.0]
err_l2_ds = ANOVAapprox.get_l2error(ads)[0.0]
err_l2_U = ANOVAapprox.get_l2error(aU)[0.0]
err_l2_rand_ds = ANOVAapprox.get_l2error(ads, X_test, y_test)[0.0]
err_l2_rand_U = ANOVAapprox.get_l2error(aU, X_test, y_test)[0.0]

println("== CHEB LSQR ==")
println("L2 ds: ", err_L2_ds)
println("L2 U: ", err_L2_U)
println("l2 ds: ", err_l2_ds)
println("l2 U: ", err_l2_U)
println("l2 rand ds: ", err_l2_rand_ds)
println("l2 rand U: ", err_l2_rand_U)

@test err_l2_ds < 0.01
@test err_l2_U < 0.01
@test err_l2_rand_ds < 0.01
@test err_l2_rand_U < 0.01
