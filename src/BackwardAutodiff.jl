module BackwardAutodiff
import Base: exp, log, sin, cos, tan, +, -, *, /, sqrt, convert, promote_rule, zero

export D

# This is the structure that stores the autodiff information.  The name is
# `Infinitesimal` spelled backward.  The elements are `x`, the result of a
# computation; `dfdy` a storage location where the derivative of some overall
# computation (`f(x)`) with respect to the output of this element is stored;
# `parent` storing the "parent" locations that have been used to produce the
# result `x`; and `bp!` a function that propogates the derivative `dfdy` to the
# parents via `dfdx = dydx * dfdy` where `dydx` is the derivative of this piece
# of the computation, `dfdy` is the derivative of the overall function with
# respect to the output, and `dfdx` is the same for the next level up.
mutable struct Lamisetinifni{T <: Number} <: Number
    x::T
    dfdy::T
    parent::Union{Lamisetinifni{T}, Array{Lamisetinifni{T},1}, Int, Nothing}
    bp!
end

function convert(::Type{Lamisetinifni{T}}, x::T) where T <: Number
    Lamisetinifni(x, zero(T), nothing, (dfdy, parents) -> nothing)
end

function convert(::Type{Lamisetinifni{T}}, x::S) where {T, S <: Number}
    Lamisetinifni(T(x), zero(T), nothing, (dfdy, parents) -> nothing)
end

function convert(::Type{Lamisetinifni{T}}, x::Lamisetinifni{T}) where T
    x
end

function zero(x::Lamisetinifni{T}) where T
    Lamisetinifni(zero(T), zero(T), nothing, (dfdy, parents) -> nothing)
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

function push_parents!(queue::Array{Lamisetinifni{T}, 1}, ::Nothing) where T
    # Do nothing
end
function push_parents!(queue::Array{Lamisetinifni{T}, 1}, i::Int) where T
    # Do nothing
end
function push_parents!(queue::Array{Lamisetinifni{T}, 1}, ls::Array{Lamisetinifni{T}, 1}) where T
    append!(queue, ls)
end
function push_parents!(queue::Array{Lamisetinifni{T}, 1}, l::Lamisetinifni{T}) where T
    push!(queue, l)
end

function backprop!(l::Lamisetinifni{T}) where T
    # Apparently we need this construction because otherwise l gets copied when
    # put into the array.
    backprop!([l])
end
function backprop!(queue::Array{Lamisetinifni{T},1}) where T
    while length(queue) > 0
        l = popfirst!(queue)
        l.bp!(l.dfdy, l.parent)
        push_parents!(queue, l.parent)
    end
end

function collect_outputs(l::Lamisetinifni{T}) where T
    queue = Lamisetinifni{T}[l]

    outputs = Lamisetinifni{T}[]

    while length(queue) > 0
        l = popfirst!(queue)
        if typeof(l.parent) <: Int
            push!(outputs, l)
        elseif typeof(l.parent) == Lamisetinifni{T}
            push!(queue, l.parent)
        elseif typeof(l.parent) == Array{Lamisetinifni{T}, 1}
            append!(queue, l.parent)
        else # Nothing
            # Do nothing
        end
    end

    outputs
end

"""    D([i], f)

Returns a function that computes the derivative of `f` (if it is
single-argument) or the gradient of `f` (if it is multi-argument or takes a
structured argument).  If `i` is given returns the `i`th component of the
gradient (though this does not reduce the cost with backprop).

Currently works only for `f` with scalar outputs.  Also, note that the autodiff
will fail unless the output type is identical to the input type (the code
automatically converts constants and non-differentiated expressions to the
appropriate type---only the input and output type of the function needs to
match).

So, for example, `D(cos)(3)` will fail (because the output type is `Float64` not
`Int`), while `D(cos)(3.0)` will work fine and return `-sin(3.0)`.

"""
function D(f)
    function dfdx(x::T) where T <: Number
        # Pass the function a backward infinitesimal whose backprop function
        # stores the backprop derivative in dfdx_store

        x = Lamisetinifni(x, zero(x), 1, (dfdy, parents) -> nothing)

        result = f(x)

        result.dfdy = one(result.x)
        backprop!(result)

        y = collect_outputs(result)[1]

        return y.dfdy
    end

    function dfdx(x::Array{T}) where T <: Number
        fargs = [Lamisetinifni(xelt, zero(xelt), i, (dfdy, parents) -> nothing) for (i, xelt) in enumerate(x)]
        result = f(fargs)
        result.dfdy = one(result.x)
        backprop!(result)
        y = collect_outputs(result)

        grad = zeros(typeof(result.x), length(x))
        for yelt in y
            grad[yelt.parent] = yelt.dfdy
        end

        return grad
    end

    function dfdx(x...)
        fargs = [Lamisetinifni(xelt, zero(xelt), i, (dfdy, parents) -> nothing) for (i, xelt) in enumerate(x)]
        result = f(fargs...)
        result.dfdy = one(result.x)
        backprop!(result)
        y = collect_outputs(result)

        grad = zeros(typeof(result.x), length(x))
        for yelt in y
            grad[yelt.parent] = yelt.dfdy
        end

        return grad
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

function bpp!(dfdy, xy)
    x, y = xy
    x.dfdy += dfdy
    y.dfdy += dfdy
end

function +(x::Lamisetinifni{T}, y::Lamisetinifni{T}) where T
    Lamisetinifni(x.x + y.x, zero(T), [x, y], bpp!)
end

function bpm!(dfdy, xy)
    x, y = xy
    x.dfdy += dfdy
    y.dfdy -= dfdy
end
function -(x::Lamisetinifni{T}, y::Lamisetinifni{T}) where T
    Lamisetinifni(x.x - y.x, zero(T), [x, y], bpm!)
end

function bpum!(dfdy, x)
    x.dfdy -= dfdy
end
function -(x::Lamisetinifni{T}) where T
    Lamisetinifni(-x.x, zero(T), x, bpum!)
end

function bpt!(dfdy, xy)
    x,y = xy
    x.dfdy += dfdy*y.x
    y.dfdy += x.x*dfdy
end
function *(x::Lamisetinifni{T}, y::Lamisetinifni{T}) where T
    Lamisetinifni(x.x*y.x, zero(T), [x,y], bpt!)
end

function /(x::Lamisetinifni{T}, y::Lamisetinifni{T}) where T
    yinv = one(T)/y.x

    function bp!(dfdy, xy)
        a,b = xy
        a.dfdy += dfdy*yinv
        b.dfdy -= a.x*dfdy*(yinv*yinv)
    end

    Lamisetinifni(x.x*yinv, zero(T), [x,y], bp!)
end

function exp(x::Lamisetinifni{T}) where T
    expx = exp(x.x)

    function bp!(dfdy, p)
        p.dfdy += dfdy*expx
    end

    Lamisetinifni(expx, zero(expx), x, bp!)
end

function sin(x::Lamisetinifni{T}) where T
    function bp!(dfdy, p)
        p.dfdy += cos(x.x)*dfdy
    end

    sx = sin(x.x)
    Lamisetinifni(sx, zero(sx), x, bp!)
end

function cos(x::Lamisetinifni{T}) where T
    function bp!(dfdy, p)
        p.dfdy -= sin(x.x)*dfdy
    end

    cx = cos(x.x)
    Lamisetinifni(cx, zero(cx), x, bp!)
end

function tan(x::Lamisetinifni{T}) where T
    c = cos(x.x)
    function bp!(dfdy, p)
        p.dfdy += dfdy/(c*c)
    end

    tx = tan(x.x)
    Lamisetinifni(tx, zero(tx), x, bp!)
end

function sqrt(x::Lamisetinifni{T}) where T
    sqrtx = sqrt(x.x)

    function bp!(dfdy, p)
        p.dfdy += dfdy/(2*sqrtx)
    end

    Lamisetinifni(sqrtx, zero(sqrtx), x, bp!)
end

function log(x::Lamisetinifni{T}) where T
    function bp!(dfdy, p)
        p.dfdy += dfdy/x.x
    end

    Lamisetinifni(log(x.x), zero(T), x, bp!)
end

end
