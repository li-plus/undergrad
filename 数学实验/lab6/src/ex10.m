U = [1000 1005 1010]';
D = [1005 1010 1015]';
U1 = U(1);
u1 = 0.8;
A = [5 5 5]';
a = [100 60 50]';
c = [1 1 1]';
s = [0.9 0.6]';
r = 1;

% x = [x1, x2, x3, u2, u3, d1, d2, d3]'
f = [c.*A; zeros(5,1)];
B = [];
b = [];
Beq = [0    0    0    1     0     -s(1) 0     0;
       0    0    0    0     1     0     -s(2) 0;
       A(1) 0    0    0     0     D(1)  0     0;
       0    A(2) 0    -U(2) 0     0     D(2)  0;
       0    0    A(3) 0     -U(3) 0     0     D(3)];
beq = [0; 0; U1*u1 + A(1)*a(1); A(2)*a(2); A(3)*a(3)];
lb = zeros(8,1);
ub = [a; r * ones(5,1)];
options = [];
[x,fval,exitflag,output] = linprog(f, B, b, Beq, beq, lb, ub, options);
