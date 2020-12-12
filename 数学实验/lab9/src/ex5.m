p = 43/50;
n = 50;
data = zeros(1,n);
data(1:round(n*p)) = 1;

mu = 0.9;
sigma = sqrt(mu*(1-mu));
[h,p,ci,zval] = ztest(data, mu, sigma, 'Tail', 'left', 'Alpha', 0.05);
h,p,ci,zval
