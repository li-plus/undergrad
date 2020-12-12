function tests = test_lagrange()
    tests = functiontests(localfunctions);
end

function test_linear(test_case)
    func = @(x) 2 * x - 1;
    x = [1, 2];
    y = func(x);
    xq = [-1, 6];
    yq = lagrange(x, y, xq);
    verifyEqual(test_case, yq, func(xq));
end

function test2(test_case)
    func = @(x) 3 .* x .^ 2 + 4 .* x - 16;
    x = [-1, 2, 3];
    y = func(x);
    xq = [-8, 6];
    yq = lagrange(x, y, xq);
    verifyEqual(test_case, yq, func(xq));
end

function test5(test_case)
    func = @(x) x.^5 + 4.*x.^4 + 3.*x.^3 + 2.*x.^2 - x + 15.0;
    x = [-4, -1, 2, 5, 7, 8];
    y = func(x);
    xq = [-3, 3];
    yq = lagrange(x, y, xq);
    verifyTrue(test_case, all(abs(yq - func(xq)) < 1e-6));
end
