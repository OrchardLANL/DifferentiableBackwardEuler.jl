using Test
@testset "DAE" begin let
	using DifferentialEquations
	import DifferentiableBackwardEuler
	DBE = DifferentiableBackwardEuler

	function rober(u, p, t)
		du = zeros(eltype(u), 3)
		du[1] = -p[1] * u[1] + p[3] * u[2] * u[3]
		du[2] = p[1] * u[1] - p[3] * u[2] * u[3] - p[2] * u[2] ^2
		du[3] = sum(u) - 1
		return du
	end
	function rober_u(u, p, t)
		du_u = zeros(eltype(u), 3, 3)
		du_u[1, :] = [-p[1], p[3] * u[3], p[3] * u[2]]
		du_u[2, :] = [p[1], -p[3] *u[3] - 2 * p[2] * u[2], -p[3] * u[2]]
		du_u[3, :] = ones(eltype(u), 3)
		return du_u
	end
	function rober_p(u, p, t)
		du_p = zeros(eltype(u), 3, 3)
		du_p[1, :] = [-u[1], 0, u[2] * u[3]]
		du_p[2, :] = [u[1], -u[2] ^2, -u[2] * u[3]]
		du_p[3, :] = zeros(eltype(u), 3)
		return du_p
	end
	rober_t(u, p, t) = zero(u)

	M = [1. 0 0; 0 1. 0; 0 0 0]
	f = ODEFunction(rober, mass_matrix=M)
	u0 = [1.0, 0.0, 0.0]
	tspan = (0.0, 1e5)
	p = [0.04, 3e7, 1e4]
	prob_mm = ODEProblem(f, u0, tspan, p)
	sol = solve(prob_mm, ImplicitEuler(), reltol=1e-8, abstol=1e-8)
	dbe_sol = Array{Float64, 1}[u0]
	#test dae_step
	for i = 1:length(sol.t) - 1
		push!(dbe_sol, DBE.dae_step(dbe_sol[end], sol.t[i + 1] - sol.t[i], rober, rober_u, rober_p, rober_t, p, sol.t[i + 1], M))
		@test isapprox(dbe_sol[end], sol.u[i + 1]; atol=1e-5, rtol=1e-5)
	end
	#test dae_steps(..., ts, M)
	dbe_sol = DBE.dae_steps(u0, rober, rober_u, rober_p, rober_t, p, sol.t, M)
    @test isapprox(dbe_sol, Matrix(sol); atol=1e-5, rtol=1e-5)
	#test dae_steps(..., t0, tfinal, M)
	dbe_sol = DBE.dae_steps(u0, rober, rober_u, rober_p, rober_t, p, tspan[1], tspan[2], M; reltol=1e-8, abstol=1e-8)
    @test isapprox(dbe_sol, Matrix(sol); atol=1e-5, rtol=1e-5)

	function naivegradient(f, x0s...; delta::Float64=1e-8)
		f0 = f(x0s...)
		gradient = map(x->zeros(size(x)), x0s)
		for j = 1:length(x0s)
			x0 = copy(x0s[j])
			for i = 1:length(x0)
				x = copy(x0)
				x[i] += delta
				xs = (x0s[1:j - 1]..., x, x0s[j + 1:length(x0s)]...)
				fval = f(xs...)
				gradient[j][i] = (fval - f0) / delta
			end
		end
		return gradient
	end
	g(p, u0) = sum(DBE.dae_steps(u0, rober, rober_u, rober_p, rober_t, p, tspan[1], 1e2, M; reltol=1e-8, abstol=1e-8)[3, end])
	dgdp_zygote = Zygote.gradient(g, p, u0)
	dgdp_naive = naivegradient(g, p, u0; delta=1e-4)
	@test isapprox(dgdp_zygote[1], dgdp_naive[1]; atol=1e-3, rtol=1e-3)
	@test isapprox(dgdp_zygote[2], dgdp_naive[2]; atol=1e-3, rtol=1e-3)
end end#end begin and let
