N = 20;
Q = 50;
q = 4.5;

fun = @(x) N * log(1 + x) + log(1 - Q / q * x);
x = 0:0.0001:0.08;
y = fun(x);
figure; plot(x, y, x, zeros(size(x)), '--'); 
xlabel('x'); ylabel('y = f(x)'); grid on;

[x, fval] = fzero(fun, 0.065);
x
fval
