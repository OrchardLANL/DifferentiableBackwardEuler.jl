using Test
@testset "DAE" begin let
	using DifferentialEquations
	import DifferentiableBackwardEuler
	DBE = DifferentiableBackwardEuler

	function rober(u,p,t)
		du = zeros(eltype(u), 3)
		du[1] = -p[1] * u[1] + p[3] * u[2] * u[3]
		du[2] = p[1] * u[1] - p[3] * u[2] * u[3] - p[2] * u[2] ^2
		du[3] = sum(u) - 1
		return du
	end
	function rober_u(u,p,t)
		du_u = zeros(eltype(u), 3, 3)
		du_u[1, :] = [-p[1], p[3] * u[3], p[3] * u[2]]
		du_u[2, :] = [p[1], -p[3] *u[3] - 2 * p[2] * u[2], -p[3] * u[2]]
		du_u[3, :] = ones(eltype(u), 3)
		return du_u
	end

	M = [1. 0 0; 0 1. 0; 0 0 0]
	f = ODEFunction(rober, mass_matrix=M)
	u0 = [1.0, 0.0, 0.0]
	tspan = (0.0, 1e5)
	p = [0.04, 3e7, 1e4]
	prob_mm = ODEProblem(f, u0, tspan, p)
	sol = solve(prob_mm, ImplicitEuler(), reltol=1e-8, abstol=1e-8)
	dbe_sol = Array{Float64, 1}[u0]
	for i = 1:length(sol.t) - 1
		push!(dbe_sol, DBE.dae_step(dbe_sol[end], sol.t[i + 1] - sol.t[i], rober, rober_u, nothing, nothing, p, sol.t[i + 1], M))
		@test isapprox(dbe_sol[end], sol.u[i + 1]; atol=1e-5, rtol=1e-5)
	end
end end#end begin and let
