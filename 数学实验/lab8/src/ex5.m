global a mu sigma;
a = 100;
sx = 80;
sy = 50;
r = 0.4;
mu = [0 0];
sigma = [sx^2  r * sx * sy; r * sx * sy  sy^2];

n = 100000;
x = unifrnd(-a, a, n, 2);
idx = x(:,1).^2 + x(:,2).^2 <= a^2;
x = x(idx, :);

p = 4 * a^2 * sum(mvnpdf(x, mu, sigma)) / n;
p

p = integral2(@pdf, -a, a, -a, a);
p

function f = pdf(x, y)
    global a mu sigma;
    idx = x.^2 + y.^2 <= a^2;
    f = zeros(size(x));
    f(idx) = mvnpdf([x(idx) y(idx)], mu, sigma);
end
