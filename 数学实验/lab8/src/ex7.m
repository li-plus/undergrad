mu = 2000;
sigma = 50;
A = 0.5;
K = 50000;
b = 0.5;
c = 0.35;

fun = @(n) normcdf(n,mu,sigma) - (2*A*n/K - A + b) / (b-c);

fzero(fun, 2000)
