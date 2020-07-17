module DifferentiableBackwardEuler

import LinearAlgebra
import NLsolve
import Zygote

#solve y1 - y0 - deltah * f(y1, p, t) = 0
function step(y0, deltah, f, f_y, f_p, f_t, p, t; kwargs...)
	function f!(F, y)
		F .= y .- y0 .- deltah .* f(y, p, t)
	end
	function j!(J, y)
		thisJ = LinearAlgebra.I - deltah * f_y(y, p, t)
		copyto!(J, thisJ)
	end
	df = NLsolve.OnceDifferentiable(f!, j!, y0, f(y0, p, t), f_y(y0, p, t))
	soln = NLsolve.nlsolve(df, y0; kwargs...)
	return soln.zero
end

@Zygote.adjoint step(y0, deltah, f, f_y, f_p, f_t, p, t; kwargs...) = begin
	y1 = step(y0, deltah, f, f_y, f_p, f_t, p, t)
	step_y0 = -LinearAlgebra.I
	step_deltah = -f(y1, p, t)
	step_p = -deltah * f_p(y1, p, t)
	step_t = -deltah * f_t(y1, p, t)
	back = delta->begin
		lambda = transpose(LinearAlgebra.I - deltah * f_y(y1, p, t)) \ delta
		return (-transpose(step_y0) * lambda,#y0
				-transpose(step_deltah) * lambda,#deltah
				nothing, nothing, nothing, nothing,#f through f_t
				-transpose(step_p) * lambda,#p
				-transpose(step_t) * lambda)#t
	end
	return y1, back
end

function steps(y0, f, f_y, f_p, f_t, p, ts; kwargs...)
	ys = Zygote.Buffer(y0, length(y0), length(ts))
	ys[:, 1] = y0
	for i = 1:length(ts) - 1
		ys[:, i + 1] = step(ys[:, i], ts[i + 1] - ts[i], f, f_y, f_p, f_t, p, ts[i + 1]; kwargs...)
	end
	return copy(ys)
end

end
