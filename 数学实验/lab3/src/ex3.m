n = 20;
A = 3*eye(n) - diag(ones(n-1,1),1)/2 - diag(ones(n-1,1), -1)/2 - ...
    diag(ones(n-2, 1), 2)/4 - diag(ones(n-2, 1), -2)/4;

b = ones(n,1);
x0 = ones(n,1);
iter_err = 1e-5;

% ground truth
x_gt = A \ b;

% decomposition of A
D = diag(diag(A));
L = -tril(A, -1);
U = -triu(A, 1);

% Jacobi
B = D \ (L + U);
f = D \ b;
[x, cnt] = solve_iter(B, f, x0, iter_err);
abs_err = max(abs(x - x_gt));
rho = max(abs(eig(B)));
fprintf('Jacobi rho: %.3f, iter: %d, abs err: %f\n', rho, cnt, abs_err);

% Gauss-Seidel
B = (D - L) \ U;
f = (D - L) \ b;
[x, cnt] = solve_iter(B, f, x0, iter_err);
abs_err = max(abs(x - x_gt));
rho = max(abs(eig(B)));
fprintf('Gauss-Seidel rho: %f, iter: %d, abs err: %f\n', rho, cnt, abs_err);

function [x, cnt] = solve_iter(B, f, x0, err)
    cnt = 0;
    x = x0;
    while 1
        x_old = x;
        x = B * x + f;
        cnt = cnt + 1;
        if max(abs(x - x_old)) < err
            break
        end
    end
end
