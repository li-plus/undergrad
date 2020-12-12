% X = [x, y, z, w, p, q, r, s]'
X0 = [450 0 0 0 0 0 0 0]';
A = [-1 -1 0 0 1 1 0 0;
     0 0 -1 0 0 0 1 1;
     0 0 0 0 1 0 1 0;
     0 0 0 0 0 1 0 1];
b = [0 0 600 200]';
lb = [0 0 0 0 0 0 0 0]';
ub = [500 500 500 inf inf inf inf inf]';
options = optimoptions('fmincon', 'Algorithm', 'sqp');
[X,fval,exitflag,output] = fmincon(@fun,X0,A,b,[],[],lb,ub,@nonlincon,options);

function [c,ceq] = nonlincon(X)
    [x,y,z,w,p,q,r,s] = decompose(X);
    ceq(1) = w * (x + y)- (3*x + y);
    c(1) = w*p + 2*r - 2.5 * (p + r);
    c(2) = w*q + 2*s - 1.5 * (q + s);
end

function [x,y,z,w,p,q,r,s] = decompose(X)
    x = X(1); y = X(2); z = X(3); w = X(4);
    p = X(5); q = X(6); r = X(7); s = X(8);
end

function z = fun(X)
    [x,y,z,w,p,q,r,s] = decompose(X);
    profit = 9*(p+r) + 15*(q+s) - 6*x - 13*y - 10*z;
    z = -profit;
end