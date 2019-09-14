# Backward Autodiff for Julia

This package implements [backward
autodiff](https://en.wikipedia.org/wiki/Automatic_differentiation#Reverse_accumulation)
for the Julia language.  It is a partner to the
[ForwardAutodiff](https://github.com/Farr-PHY-604/ForwardAutodiff.jl) package,
coded as part of the PHY 604: Computational Methods in Physics and Astrophysics
II course.  

If you have written a function (of any number of arguments---or which takes an
array as input) than produces a scalar output, and you want to have derivatives
or gradients of this function efficiently and automatically, this package is for
you.  The only exported function is `D`, which maps functions to their
derivative (or their gradient in the case of multiple inputs).  For example:

```julia
julia> using BackwardAutodiff
julia> r = (x,y,z) -> sqrt(x*x + y*y + z*z)
julia> r_hat = D(r)
julia> r_hat(1,1,1)
3-element Array{Float64,1}:
 0.5773502691896258
 0.5773502691896258
 0.5773502691896258
julia> [1,1,1]/sqrt(3)
3-element Array{Float64,1}:
 0.5773502691896258
 0.5773502691896258
 0.5773502691896258
```

See instructions for the ForwardAutodiff.jl package for more description and a
tutorial notebook about autodiff as well as notes about how to use packages in
general for Julia.  
