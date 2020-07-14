using Test
import DifferentiableBackwardEuler
DBE = DifferentiableBackwardEuler
import SparseArrays
import Zygote

f(u, p, t) = f_u(u, p, t) * u
f_u(u, p, t) = SparseArrays.spdiagm(0=>p)
f_p(u, p, t) = SparseArrays.spdiagm(0=>u)
f_t(u, p, t) = zero(u)

n = 2
p = collect(1.0:n)
u0 = ones(n)
us = [u0]
numsteps = 10 ^ 3
ts = range(0, 1; length=numsteps)
for i = 1:length(ts) - 1
	push!(us, DBE.step(us[end], ts[i + 1] - ts[i], f, f_u, f_p, f_t, p, ts[i + 1]))
end
@test isapprox(exp.(p), us[end]; atol=1e2 / numsteps, atol=1e2 / numsteps)

numsteps = 10 ^ 3
ts = range(0, 1e0; length=numsteps)
function g(p, u0)
	us = [u0]
	us = Zygote.Buffer(u0, length(u0), numsteps)
	us[:, 1] = u0
	for i = 1:length(ts) - 1
		us[:, i + 1] = DBE.step(us[:, i], ts[i + 1] - ts[i], f, f_u, f_p, f_t, p, ts[i + 1])
	end
	return sum(us[:, numsteps])
end
@test isapprox(sum(exp.(p * ts[end])), g(p, u0); atol=1e2 / numsteps, rtol=1e2 / numsteps)

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

dgdp_zygote = Zygote.gradient(g, p, u0)
dgdp_naive = naivegradient(g, p, u0)
@test isapprox(dgdp_zygote[1], dgdp_naive[1]; atol=1e-4, rtol=1e-4)
@test isapprox(dgdp_zygote[2], dgdp_naive[2]; atol=1e-4, rtol=1e-4)
