credit = [2 2 1 1 5]';
due = [9 15 4 3 2]';
profit = [0.043 0.054 0.050 0.044 0.045]';
tax = [0 0.5 0.5 0.5 0]';

f = (1-tax) .* profit;

A = [0 -1 -1 -1 0;
    credit' - 1.4;
    due' - 5;
    ones(1,5)];
b = [-400 0 0 1000]';

[x,fval,exitflag,output,lambda] = ...
    linprog(-f, A, b, [], [], zeros(5,1), []);
x, fval

f = [f; -0.0275];
A = [A [0 0 0 -1]'];
[x,fval,exitflag,output,lambda] = ...
    linprog(-f, A, b, [], [], zeros(6,1), [inf*ones(5,1); 100]);
x, fval
