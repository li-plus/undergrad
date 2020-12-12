Q = [1.0 0.192 2.169 1.611;
    0.316 1.0 0.477 0.524;
    0.377 0.360 1.0 0.296;
    0.524 0.282 2.065 1.0];
n = 4;
a = [18.607 15.841 20.443 19.293]';
b = [3643.31 2755.64 4628.96 4117.07]';
c = [239.73 219.16 252.64 227.44]';
P = 760;

% search and solve
results = zeros(4, 2000);
results_cnt = 0;
for x1 = 0:0.1:1
    for x2 = 0:0.1:1-x1
        for x3 = 0:0.1:1-x1-x2
            for T = 0:10:100
                XT0 = [x1 x2 x3 T]';
                [XT, val] = fsolve(@eqn, XT0, [], n, P, a, b, c, Q);
                x = [XT(1:n-1); 1 - sum(XT(1:n-1))];
                results_cnt = results_cnt + 1;
                results(:, results_cnt) = XT';
            end
        end
    end
end

% filter results
results = real(results);
valid_results = results(:, 1);
valid_cnt = 1;

for result = results
    if min(sum(abs(result - valid_results), 1)) > 1e-4
        if all(result(1:3) > -1e-5) && sum(result(1:3)) < 1 + 1e-5
            valid_cnt = valid_cnt + 1;
            valid_results(:, valid_cnt) = result;
        end
    end
end

% error analysis
err = zeros(4, valid_cnt);
err_1 = zeros(1, valid_cnt);
err_2 = zeros(1, valid_cnt);
err_inf = zeros(1, valid_cnt);
for i = 1:valid_cnt
    err(:, i) = eqn(valid_results(:, i), n, P, a, b, c, Q);
    err_1(i) = norm(err(:, i), 1);
    err_2(i) = norm(err(:, i), 2);
    err_inf(i) = norm(err(:, i), inf);
end

% equation to solve
function y = eqn(XT, n, P, a, b, c, Q)
    x = [XT(1:n-1); 1 - sum(XT(1:n-1))];
    T = XT(n);
    Qx = Q * x;
    y = x .* (1 - log(Qx) - Q' * (x ./ Qx) + a - b ./ (T + c) - log(P));
end
