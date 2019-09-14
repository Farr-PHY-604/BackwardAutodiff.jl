module BackwardAutodiff
import Base: exp, sin, cos, tan, +, -, *, /, sqrt, convert, promote_rule, zero

export D

# This takes some explanation.  The method here is for each operation to compute
# a result *and* provide a function for later use that will perform the
# backpropogation.  For an operation that maps x -> y in the contex of an
# overall computation of some function f, the backprop function will be provided
# with the value of df/dy, and then should compute df/dx = df/dy * dy/dx (i.e.
# multiply the provided value by the derivative of the x->y computation) and
# provide df/dx to the bp functions of its inputs---that is, to propogate it "up
# the chain."
struct Lamisetinifni{T <: Number} <: Number
    x::T
    bp
end

function convert(::Type{Lamisetinifni{T}}, x::T) where T <: Number
    Lamisetinifni(x, dfdy -> nothing)
end

function convert(::Type{Lamisetinifni{T}}, x::Lamisetinifni{S}) where {T, S}
    Lamisetinifni(T(x.x), x.bp)
end

function convert(::Type{Lamisetinifni{T}}, x::S) where {T, S <: Number}
    Lamisetinifni(T(x), dfdy -> nothing)
end

function zero(x::Lamisetinifni{T}) where T
    Lamisetinifni(zero(x.x), dfdy -> nothing)
end

function promote_rule(::Type{Lamisetinifni{T}}, ::Type{Lamisetinifni{S}}) where {T, S}
    Lamisetinifni{promte_type(T, S)}
end

function promote_rule(::Type{Lamisetinifni{T}}, ::Type{S}) where {T, S <: Number}
    Lamisetinifni{promote_type(T,S)}
end

function promote_rule(::Type{T}, ::Type{Lamisetinifni{S}}) where {T <: Number, S}
    Lamisetinifni{promote_type(T,S)}
end

function promote_rule(::Type{S}, ::Type{Lamisetinifni{T}}) where {S <: AbstractIrrational, T}
    Lamisetinifni{promote_type(S, T)}
end

"""    D([i], f)

Returns a function that computes the derivative of `f` (if it is
single-argument) or the gradient of `f` (if it is multi-argument or takes a
structured argument).  If `i` is given returns the `i`th component of the
gradient (though this does not reduce the cost with backprop).
"""
function D(f)
    function dfdx(x::T) where T <: Number
        dfdx_store = zero(x)

        # Pass the function a backward infinitesimal whose backprop function
        # stores the backprop derivative in dfdx_store
        result = f(Lamisetinifni(x, dfdy -> dfdx_store += dfdy))

        # Call the backprop function that is returned from the evaluation of f
        # with derivative one.  That function will backprop "up" the stack of f,
        # with the result that the accumulated gradient will be stored in
        # dfdx_store.
        result.bp(one(result.x))

        return dfdx_store
    end

    function dfdx(x::Array{T}) where T <: Number
        dfdx_store = zeros(T, size(x)...)

        fargs = [Lamisetinifni(xelt, dfdy -> dfdx_store[i] += dfdy) for (i, xelt) in enumerate(x)]
        result = f(fargs)
        result.bp(one(result.x))

        return dfdx_store
    end

    function dfdx(x...)
        dfdx_store = [zero(xelt) for xelt in x]

        fargs = [Lamisetinifni(xelt, dfdy -> dfdx_store[i] += dfdy) for (i, xelt) in enumerate(x)]
        result = f(fargs...)
        result.bp(one(result.x))

        return dfdx_store
    end

    return dfdx
end

function D(i::Integer, f)
    df = D(f)
    function df_wrapper(x...)
        g = df(x...)
        return g[i]
    end
    return df_wrapper
end

function +(x::Lamisetinifni{T}, y::Lamisetinifni{T}) where T
    # Unit derivative, so dfdy backprops to both arguments.
    function bp(dfdy)
        x.bp(dfdy)
        y.bp(dfdy)
    end

    Lamisetinifni(x.x + y.x, bp)
end

function -(x::Lamisetinifni{T}, y::Lamisetinifni{T}) where T
    function bp(dfdy)
        x.bp(dfdy)
        y.bp(-dfdy)
    end

    Lamisetinifni(x.x - y.x, bp)
end
function -(x::Lamisetinifni{T}) where T
    Lamisetinifni(-x.x, dfdy -> x.bp(-dfdy))
end

function *(x::Lamisetinifni{T}, y::Lamisetinifni{T}) where T
    function bp(dfdy)
        x.bp(y.x*dfdy)
        y.bp(x.x*dfdy)
    end

    Lamisetinifni(x.x*y.x, bp)
end

function /(x::Lamisetinifni{T}, y::Lamisetinifni{T}) where T
    function bp(dfdy)
        x.bp(dfdy/y.x)
        y.bp(-x.x*dfdy/(y.x*y.x))
    end

    Lamisetinifni(x.x/y.x, bp)
end

function sqrt(x::Lamisetinifni{T}) where T
    sqrtx = sqrt(x.x)
    Lamisetinifni(sqrtx, dfdy -> x.bp(one(x.x)/(2*sqrtx)*dfdy))
end

function exp(x::Lamisetinifni{T}) where T
    expx = exp(x.x)
    Lamisetinifni(expx, dfdy -> x.bp(expx*dfdy))
end

function sin(x::Lamisetinifni{T}) where T
    Lamisetinifni(sin(x.x), dfdy -> x.bp(cos(x.x)*dfdy))
end

function cos(x::Lamisetinifni{T}) where T
    Lamisetinifni(cos(x.x), dfdy -> x.bp(-sin(x.x)*dfdy))
end

function tan(x::Lamisetinifni{T}) where T
    c = cos(x.x)
    Lamisetinifni(tan(x.x), dfdy -> x.bp(dfdy/(c*c)))
end

end
