b = [0 0 5 3 0];
s = [0.4 0.6 0.6 0.4];
h = [0 500 400 200 100]';
n = length(b);

L = [b; diag(s) zeros(n-1, 1)];
I = eye(n);
x = (L - I) \ h
err = (L - I) * x - h
cond(L - I)
