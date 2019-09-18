using BackwardAutodiff
using Test: @test, @testset

@testset "FowardAutodiff.jl tests" begin
    @testset "Basic arithmetic" begin
        x = randn()
        y = randn()
        z = randn()

        @test isapprox(D(x->2*x)(x), 2)
    end

    @testset "Transcendental Functions" begin
        x = randn()
        @test isapprox(D(exp)(x), exp(x))
        @test isapprox(D(sin)(x), cos(x))
        @test isapprox(D(cos)(x), -sin(x))
        @test isapprox(D(tan)(x), sec(x)*sec(x))
    end

    @testset "sqrt" begin
        x = randn()

        # We need to be in the complex domain explicitly, or sqrt(x < 0) will error out
        @test isapprox(D(sqrt)(x + 0im), 1/(2*sqrt(x + 0im)))
    end

    @testset "compound function bits" begin
        x = randn()
        y = randn()
        @test isapprox(D(x -> x*y)(x), y)
        @test isapprox(D(x -> x/y)(x), 1/y)
        @test isapprox(D(y -> x/y)(y), -x/(y*y))
        @test isapprox(D(x -> exp(x/y))(x), exp(x/y)/y)
        @test isapprox(D(x -> sin(x*y))(x), y*cos(x*y))
        @test isapprox(D(x -> cos(x*y))(x), -y*sin(x*y))
        @test isapprox(D(x -> x-y)(x), 1)
        @test isapprox(D(y -> x-y)(y), -1)
        @test isapprox(D(y -> x*y)(y), x)
        @test isapprox(D(x -> -x)(x), -1)
        @test isapprox(D(x -> exp(-x/pi))(x), -exp(-x/pi)/pi)
        @test isapprox(D(x -> sin(2*pi*sqrt(2)*x))(x), 2*pi*sqrt(2)*cos(2*pi*sqrt(2)*x))
        @test isapprox(D(x -> exp(-x/pi))(3.0), -exp(-3/pi)/pi)
        @test isapprox(D(x -> exp(-x)*sin(x))(x), -exp(-x)*sin(x) + exp(-x)*cos(x))
        @test isapprox(D(x -> exp(-x))(x), -exp(-x))
        @test isapprox(D(x -> x*x)(x), 2*x)
    end

    @testset "compound function" begin
        freq = sqrt(2)
        A = randn()
        B = randn()

        a = A/sqrt(A*A + B*B)
        b = B/sqrt(A*A + B*B)
        phi = atan(b, a)

        x = randn()

        function f(x)
            return exp(-x/pi)*(b*cos(2*pi*freq*x) + a*sin(2*pi*freq*x))
        end
        fprime = D(f)

        function laborious_fprime(x)
            return exp(-x/pi)*(2*pi*freq*cos(2*pi*freq*x + phi) - sin(2*pi*freq*x + phi)/pi)
        end

        @test isapprox(fprime(x), laborious_fprime(x))
    end

    @testset "partials of simple function" begin
        function linear(x, m, b)
            return m*x+b
        end
        dldx = D(1, linear)
        dldslope = D(2, linear)
        dldintercept = D(3, linear)

        x = randn()
        m = randn()
        b = randn()
        @test isapprox(dldx(x,m,b), m)
        @test isapprox(dldslope(x,m,b), x)
        @test isapprox(dldintercept(x,m,b), 1)
    end

    @testset "gradient of many-arg and vector function" begin
        function linear(x, m, b)
            m*x + b
        end

        function vlinear(xmb)
            linear(xmb...)
        end

        g = D(linear)
        vg = D(vlinear)

        x = randn()
        m = randn()
        b = randn()

        @test isapprox(g(x, m, b), [m, x, 1])
        @test isapprox(vg([x,m,b]), [m, x, 1])
    end

    @testset "multivariate polynomial" begin
        coeffs = randn(5)
        function p(x,y,z)
            coeffs[1] + x*(coeffs[2] + y*(coeffs[3] + z*(coeffs[4] + x*coeffs[5])))
        end

        function gp(x,y,z)
            [coeffs[2] + y*(coeffs[3] + z*(coeffs[4] + x*coeffs[5])) + x*y*z*coeffs[5],
             x*(coeffs[3] + z*(coeffs[4] + x*coeffs[5])),
             x*y*(coeffs[4] + x*coeffs[5])]
         end

         x,y,z = randn(3)

         @test isapprox(D(p)(x,y,z), gp(x,y,z))
    end

    @testset "log derivative" begin
        x = rand()
        @test isapprox(D(log)(x), 1/x)

        @test isapprox(D(x -> log(exp(x)))(x), 1.0)
    end
end
