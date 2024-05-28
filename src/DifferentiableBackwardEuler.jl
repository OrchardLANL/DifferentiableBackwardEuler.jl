module DifferentiableBackwardEuler

import DifferentialEquations
import ChainRulesCore
import LinearAlgebra
import NLsolve

function nlsolve_solver(f!, j!, F0, J0, y0; kwargs...)
	df = NLsolve.OnceDifferentiable(f!, j!, y0, F0, J0)
	soln = NLsolve.nlsolve(df, y0; kwargs...)
	if !NLsolve.converged(soln)
		@show soln
		@show t
		error("solution did not converge")
	end
	return soln.zero
end

function backslash_solver(f!, j!, F0, J0, y0; kwargs...)
    return J0 \ -F0
end

#solve M * (y1 - y0) - deltah * f(y1, p, t) = 0
function dae_step(y0, deltah, f, f_y, f_p, f_t, p, t, M; dbe_solver=nlsolve_solver, kwargs...)
	function f!(F, y)
		myF = M * (y .- y0) .- deltah * f(y, p, t)
		copy!(F, myF)
	end
	function j!(J, y)
		thisJ = M - deltah * f_y(y, p, t)
		copyto!(J, thisJ)
	end
	J0 = f_y(y0, p, t)
	j!(J0, y0)
	F0 = f(y0, p, t)
	f!(F0, y0)
    return dbe_solver(f!, j!, F0, J0, y0; kwargs...)
end

#solve y1 - y0 - deltah * f(y1, p, t) = 0
function step(y0, deltah, f, f_y, f_p, f_t, p, t; kwargs...)
	return dae_step(y0, deltah, f, f_y, f_p, f_t, p, t, LinearAlgebra.I; kwargs...)
end

function make_dae_step_pullback(y1, y0, deltah, f, f_y, f_p, f_t, p, t, M)
	function dae_step_pullback(delta)
		step_y0 = -M
		step_deltah = -f(y1, p, t)
		step_p = -deltah * f_p(y1, p, t)
		step_t = -deltah * f_t(y1, p, t)
		lambda = transpose(M - deltah * f_y(y1, p, t)) \ delta
		return (ChainRulesCore.NoTangent(),#step function
				@ChainRulesCore.thunk(-transpose(step_y0) * lambda),#y0
				@ChainRulesCore.thunk(-transpose(step_deltah) * lambda),#deltah
				ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),#f through f_t
				@ChainRulesCore.thunk(-transpose(step_p) * lambda),#p
				@ChainRulesCore.thunk(-transpose(step_t) * lambda),#t
				ChainRulesCore.NoTangent())#M -- this is wrong but we're not supporting differentation w.r.t. M
	end
	return dae_step_pullback
end

function ChainRulesCore.rrule(::typeof(dae_step), y0, deltah, f, f_y, f_p, f_t, p, t, M; kwargs...)
	y1 = dae_step(y0, deltah, f, f_y, f_p, f_t, p, t, M; kwargs...)
	pullback = make_dae_step_pullback(y1, y0, deltah, f, f_y, f_p, f_t, p, t, M)
	return y1, pullback
end

function steps(y0, f, f_y, f_p, f_t, p, ts; kwargs...)
	return dae_steps(y0, f, f_y, f_p, f_t, p, ts, LinearAlgebra.I; kwargs...)
end

function dae_steps(y0, f, f_y, f_p, f_t, p, ts, M; kwargs...)
	ys = zeros(eltype(y0), length(y0), length(ts))
	ys[:, 1] = y0
	for i = 1:length(ts) - 1
		ys[:, i + 1] = dae_step(ys[:, i], ts[i + 1] - ts[i], f, f_y, f_p, f_t, p, ts[i + 1], M; kwargs...)
	end
	return ys
end

function make_dae_steps_pullback(y, y0, f, f_y, f_p, f_t, p, ts, M)
	dae_step_pullbacks = [make_dae_step_pullback(y[:, i], y[:, i - 1], ts[i] - ts[i - 1], f, f_y, f_p, f_t, p, ts[i], M) for i = 2:size(y, 2)]
	function dae_steps_pullback(delta)
		_, thisdy0, _, _, _, _, _, thisdp, _ = dae_step_pullbacks[size(y, 2) - 1](delta[:, size(y, 2)])
		dy0 = ChainRulesCore.unthunk(thisdy0) .+ delta[:, size(y, 2) - 1]
		dp = ChainRulesCore.unthunk(thisdp)
		for i = reverse(1:size(y, 2) - 2)
			_, thisdy0, _, _, _, _, _, thisdp, _ = dae_step_pullbacks[i](dy0)
			dy0 .= ChainRulesCore.unthunk(thisdy0) .+ delta[:, i]
			dp .= ChainRulesCore.unthunk(thisdp) .+ dp
		end
		return (ChainRulesCore.NoTangent(),#steps
				dy0,#y0
				ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),#f through f_t
				dp,#p
				ChainRulesCore.ZeroTangent(),#ts -- just make it zero for now
				ChainRulesCore.NoTangent())#M -- again this is wrong, but we basically don't support this for now
	end
	return dae_steps_pullback
end

function ChainRulesCore.rrule(::typeof(dae_steps), y0, f, f_y, f_p, f_t, p, ts, M; kwargs...)
	y = dae_steps(y0, f, f_y, f_p, f_t, p, ts, M; kwargs...)
	dae_steps_pullback = make_dae_steps_pullback(y, y0, f, f_y, f_p, f_t, p, ts, M)
	return y, dae_steps_pullback
end

function steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal; kwargs...)
	return dae_steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal, LinearAlgebra.I; kwargs...)
end
function dae_steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal, M; soln_callback=soln->nothing, kwargs...)
	odef = DifferentialEquations.ODEFunction(f; jac=f_y, jac_prototype=f_y(y0, p, t0), mass_matrix=M)
	prob = DifferentialEquations.ODEProblem(odef, y0, (t0, tfinal), p)
	soln = DifferentialEquations.solve(prob, DifferentialEquations.ImplicitEuler(); kwargs...)
    soln_callback(soln)
	return soln
end
function steps(y0, f, f_y, f_p, f_t, p, t0, tfinal; kwargs...)
	return dae_steps(y0, f, f_y, f_p, f_t, p, t0, tfinal, LinearAlgebra.I; kwargs...)
end
function dae_steps(y0, f, f_y, f_p, f_t, p, t0, tfinal, M; kwargs...)
    return Matrix(dae_steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal, M; kwargs...))
end

function make_dae_steps_pullback(y, y0, f, f_y, f_p, f_t, p, t0, tfinal, ts, M)
	pullback1 = make_dae_steps_pullback(y, y0, f, f_y, f_p, f_t, p, ts, M)
	function dae_steps_pullback(delta)
		_, dy0, _, _, _, _, dp, _ = pullback1(delta)
		return (ChainRulesCore.NoTangent(),#steps
				dy0,#y0
				ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(),#f through f_t
				dp,#p
				ChainRulesCore.ZeroTangent(),#t0 -- just make it zero for now
				ChainRulesCore.ZeroTangent(),#tfinal -- just make it zero for now
				ChainRulesCore.NoTangent())#M -- this is incorrect but differentiating w.r.t. M is no supported
	end
	return dae_steps_pullback
end

function ChainRulesCore.rrule(::typeof(dae_steps), y0, f, f_y, f_p, f_t, p, t0, tfinal, M; kwargs...)
	soln = dae_steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal, M; kwargs...)
    y = Matrix(soln)
	steps_pullback = make_dae_steps_pullback(y, y0, f, f_y, f_p, f_t, p, t0, tfinal, soln.t, M)
	return y, steps_pullback
end

end
