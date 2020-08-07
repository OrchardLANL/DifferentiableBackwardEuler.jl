module DifferentiableBackwardEuler

import DifferentialEquations
import ChainRulesCore
import LinearAlgebra
import NLsolve

#solve M * (y1 - y0) - deltah * f(y1, p, t) = 0
function dae_step(y0, deltah, f, f_y, f_p, f_t, p, t, M; kwargs...)
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
	df = NLsolve.OnceDifferentiable(f!, j!, y0, F0, J0)
	soln = NLsolve.nlsolve(df, y0; kwargs...)
	if !NLsolve.converged(soln)
		@show soln
		@show t
		error("solution did not converge")
	end
	return soln.zero

end

#solve y1 - y0 - deltah * f(y1, p, t) = 0
function step(y0, deltah, f, f_y, f_p, f_t, p, t; kwargs...)
	function f!(F, y)
		F .= y .- y0 .- deltah .* f(y, p, t)
	end
	function j!(J, y)
		thisJ = LinearAlgebra.I - deltah * f_y(y, p, t)
		copyto!(J, thisJ)
	end
	J0 = f_y(y0, p, t)
	j!(J0, y0)
	F0 = f(y0, p, t)
	f!(F0, y0)
	df = NLsolve.OnceDifferentiable(f!, j!, y0, F0, J0)
	soln = NLsolve.nlsolve(df, y0; kwargs...)
	if !NLsolve.converged(soln)
		@show soln
		@show t
		error("solution did not converge")
	end
	return soln.zero
end

function make_step_pullback(y1, y0, deltah, f, f_y, f_p, f_t, p, t)
	function step_pullback(delta)
		step_y0 = -LinearAlgebra.I
		step_deltah = -f(y1, p, t)
		step_p = -deltah * f_p(y1, p, t)
		step_t = -deltah * f_t(y1, p, t)
		lambda = transpose(LinearAlgebra.I - deltah * f_y(y1, p, t)) \ delta
		return (ChainRulesCore.NO_FIELDS,#step function
				@ChainRulesCore.thunk(-transpose(step_y0) * lambda),#y0
				@ChainRulesCore.thunk(-transpose(step_deltah) * lambda),#deltah
				ChainRulesCore.NO_FIELDS, ChainRulesCore.NO_FIELDS, ChainRulesCore.NO_FIELDS, ChainRulesCore.NO_FIELDS,#f through f_t
				@ChainRulesCore.thunk(-transpose(step_p) * lambda),#p
				@ChainRulesCore.thunk(-transpose(step_t) * lambda))#t
	end
	return step_pullback
end

function ChainRulesCore.rrule(::typeof(step), y0, deltah, f, f_y, f_p, f_t, p, t; kwargs...)
	y1 = step(y0, deltah, f, f_y, f_p, f_t, p, t; kwargs...)
	pullback = make_step_pullback(y1, y0, deltah, f, f_y, f_p, f_t, p, t)
	return y1, pullback
end

function steps(y0, f, f_y, f_p, f_t, p, ts; kwargs...)
	ys = zeros(eltype(y0), length(y0), length(ts))
	ys[:, 1] = y0
	for i = 1:length(ts) - 1
		ys[:, i + 1] = step(ys[:, i], ts[i + 1] - ts[i], f, f_y, f_p, f_t, p, ts[i + 1]; kwargs...)
	end
	return ys
end

function make_steps_pullback(y, y0, f, f_y, f_p, f_t, p, ts)
	step_pullbacks = [make_step_pullback(y[:, i], y[:, i - 1], ts[i] - ts[i - 1], f, f_y, f_p, f_t, p, ts[i]) for i = 2:size(y, 2)]
	function steps_pullback(delta)
		_, thisdy0, _, _, _, _, _, thisdp, _ = step_pullbacks[size(y, 2) - 1](delta[:, size(y, 2)])
		dy0 = ChainRulesCore.unthunk(thisdy0) .+ delta[:, size(y, 2) - 1]
		dp = ChainRulesCore.unthunk(thisdp)
		for i = reverse(1:size(y, 2) - 2)
			_, thisdy0, _, _, _, _, _, thisdp, _ = step_pullbacks[i](dy0)
			dy0 .= ChainRulesCore.unthunk(thisdy0) .+ delta[:, i]
			dp .= ChainRulesCore.unthunk(thisdp) .+ dp
		end
		return (ChainRulesCore.NO_FIELDS,#steps
				dy0,#y0
				ChainRulesCore.NO_FIELDS, ChainRulesCore.NO_FIELDS, ChainRulesCore.NO_FIELDS, ChainRulesCore.NO_FIELDS,#f through f_t
				dp,#p
				ChainRulesCore.Zero())#ts -- just make it zero for now
	end
	return steps_pullback
end

function ChainRulesCore.rrule(::typeof(steps), y0, f, f_y, f_p, f_t, p, ts; kwargs...)
	y = steps(y0, f, f_y, f_p, f_t, p, ts; kwargs...)
	steps_pullback = make_steps_pullback(y, y0, f, f_y, f_p, f_t, p, ts)
	return y, steps_pullback
end

function steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal; kwargs...)
	odef = DifferentialEquations.ODEFunction(f; jac=f_y, jac_prototype=f_y(y0, p, t0))
	prob = DifferentialEquations.ODEProblem(odef, y0, (t0, tfinal), p)
	soln = DifferentialEquations.solve(prob, DifferentialEquations.ImplicitEuler(); kwargs...)
	return soln
end
function steps(y0, f, f_y, f_p, f_t, p, t0, tfinal; kwargs...)
	return steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal; kwargs...)[:, :]
end

function make_steps_pullback(y, y0, f, f_y, f_p, f_t, p, t0, tfinal, ts)
	pullback1 = make_steps_pullback(y, y0, f, f_y, f_p, f_t, p, ts)
	function steps_pullback(delta)
		_, dy0, _, _, _, _, dp, _ = pullback1(delta)
		return (ChainRulesCore.NO_FIELDS,#steps
				dy0,#y0
				ChainRulesCore.NO_FIELDS, ChainRulesCore.NO_FIELDS, ChainRulesCore.NO_FIELDS, ChainRulesCore.NO_FIELDS,#f through f_t
				dp,#p
				ChainRulesCore.Zero(),#t0 -- just make it zero for now
				ChainRulesCore.Zero())#tfinal -- just make it zero for now
	end
	return steps_pullback
end

function ChainRulesCore.rrule(::typeof(steps), y0, f, f_y, f_p, f_t, p, t0, tfinal; kwargs...)
	soln = steps_diffeq(y0, f, f_y, f_p, f_t, p, t0, tfinal; kwargs...)
	y = soln[:, :]
	steps_pullback = make_steps_pullback(y, y0, f, f_y, f_p, f_t, p, t0, tfinal, soln.t)
	return y, steps_pullback
end

end
