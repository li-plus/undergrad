sigma = 0.2;
l = 2;

m = 2:0.01:3;
func_u = @(m) m - l.*(1-normcdf(l,m,sigma));
func_v = @(m) m ./ (1-normcdf(l,m,sigma)) - l;

u = func_u(m);
v = func_v(m);
plot(m, u, m, v);
xlabel('m'); ylabel('waste'); grid on;
legend('u(m)', 'v(m)');

[m, fval] = fminunc(func_u, 2.3);
m, fval
[m, fval] = fminunc(func_v, 2.3);
m, fval
