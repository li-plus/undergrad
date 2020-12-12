circle_func = @(x) sqrt(1 - x .* x);
error_limit = 1e-8;

lo_cnt = 1;
hi_cnt = 10000000;
while 1
    if (hi_cnt - lo_cnt < 2)
        break
    end
    fcnt = floor((lo_cnt + hi_cnt) / 2);
    x = linspace(0, 1, fcnt);
    y = circle_func(x);
    pi_result = 4 * trapz(x, y);
    err = abs(pi_result - pi);
    if (err < error_limit)
        hi_cnt = fcnt;
    else
        lo_cnt = fcnt;
    end
end
fprintf('ans: %.12f, cost: %d, error: %.2e\n', pi_result, fcnt, err);

int_funcs = {@quad, @quadl};
for i = 1:length(int_funcs)
    int_func = int_funcs{i};
    lo_tol = 1e-11;
    hi_tol = 1e-1;
    while 1
        if (abs(hi_tol - lo_tol) < 1e-11)
            break
        end
        tol = sqrt(hi_tol * lo_tol);
        [area, fcnt] = int_func(circle_func, 0, 1, tol);
        pi_result = 4 * area;
        err = abs(pi_result - pi);
        if (err < error_limit)
            lo_tol = tol;
        else
            hi_tol = tol;
        end
    end
    fprintf('ans: %.12f, cost: %d, error: %.2e\n', pi_result, fcnt, err);
end
