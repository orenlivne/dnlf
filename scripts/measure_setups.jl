using DNLF, LinearAlgebra
net = load_tntp_net(joinpath(pkgdir(DNLF),"data","SiouxFalls","SiouxFalls_net.tntp"))
od  = load_tntp_trips(joinpath(pkgdir(DNLF),"data","SiouxFalls","SiouxFalls_trips.tntp"))
# aggregate balanced demand (single fungible commodity), as in the §5.3 single-commodity design
d = zeros(net.n); for i in 1:size(od,1), j in 1:size(od,2); d[i]+=od[i,j]; d[j]-=od[i,j]; end
# single-OD correctness demand (as §5.1 solve_ue), pick largest OD
r,s = Tuple(argmax(od)); D = od[r,s]
d1 = zeros(net.n); d1[r]=D; d1[s]=-D
for (lbl,dem) in (("single-OD (§5.1)",d1), ("aggregate (§5.3)",d))
  _,_,steps,setups = DNLF.solve_flow(net, dem, zeros(net.m); inner=:multigrid, tol=1e-9)
  println("$lbl : Newton steps=$steps  AMG setups=$setups")
end
