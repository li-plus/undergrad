global A mu K sigma b c;
mu = 2000;
sigma = 50;
A = 0.5;
K = 50000;
b = 0.5;
c = 0.35;

n = 0:15000;
y = profit(n);
plot(n, y);
grid on; xlabel('n'); ylabel('V(n)');

function out = profit(n)
    global A mu K sigma b c;
    out = zeros(size(n));
    for i = 1:length(n)
        ni = n(i);
        a = A*(1-ni./K);
        fun = @(r) ((b-a).*r-(a-c).*(ni-r)).*normpdf(r,mu,sigma);
        out(i) = integral(fun,0,ni) + (b-a)*ni*(1-normcdf(ni));
    end
end
