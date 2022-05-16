DifferentiableBackwardEuler.jl: Automatically differentiating the backward Euler method
===============================

Description
-----------

DifferentiableBackwardEuler.jl is a [Julia](http://julialang.org/) module that automatically differentiates the [backward Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method). It is used in the [DPFEHM](https://github.com/OrchardLANL/DPFEHM.jl) subsurface flow simulator. Here are a couple illustartive examples:

```julia
import DifferentiableBackwardEuler
import SparseArrays
import Zygote

f(u, p, t) = f_u(u, p, t) * u
f_u(u, p, t) = SparseArrays.spdiagm(0=>p)
f_p(u, p, t) = SparseArrays.spdiagm(0=>u)
f_t(u, p, t) = zero(u)

numsteps = 10 ^ 2
n = 2
u0 = ones(n)
p = collect(1.0:n)
ts = collect(range(0, 1e0; length=numsteps))
g(p, u0) = sum(DifferentiableBackwardEuler.steps(u0, f, f_u, f_p, f_t, p, ts)[:, end])
g(p, u0)#≈sum(exp.(p * ts[end])) --  integrates the ODE using the backward Euler method using the ts to define the time steps
#we can compute the gradient of g using Zygote.gradient
dgdp_zygote = Zygote.gradient((p, u0)->g(p, u0), p, u0)

#you can also solve the ODE specifying only the initial and final times, and this falls back to DifferentialEquations.jl's ImplicitEuler() method to do adaptive time-stepping
t0 = 0.0
tfinal = 1e0
g(p, u0) = sum(DifferentiableBackwardEuler.steps(u0, f, f_u, f_p, f_t, p, t0, tfinal; reltol=1e-4)[:, end])#the kwargs get passed to DifferentialEquations.solve(...;kwargs...)
g(p, u0)#≈sum(exp.(p * tfinal)) --  integrates the ODE using the backward Euler method from t0 to tfinal
#again, we can compute the gradient of g using Zygote.gradient
dgdp_zygote = Zygote.gradient((p, u0)->g(p, u0), p, u0)
```

License
-------

DifferentiableBackwardEuler is provided under a BSD style license. See LICENSE.md file for the full text.

This package is part of the Orchard suite, known internally as C20086 Orchard.

Author
------

Daniel O'Malley, <omalled@lanl.gov>
