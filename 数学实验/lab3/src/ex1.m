perturb_b = 1;

for n = [5,7,9]
    for eps = [1e-10, 1e-8, 1e-6]
        A1 = fliplr(vander(1 + 0.1 * (0:n-1)));
        A2 = hilb(n);
        b1 = sum(A1, 2);
        b2 = sum(A2, 2);
        % back up
        org_A1 = A1;
        org_A2 = A2;
        org_b1 = b1;
        org_b2 = b2;
        % perturbation
        if perturb_b
            b1(n) = b1(n) + eps;
            b2(n) = b2(n) + eps;
        else
            A1(n,n) = A1(n,n) + eps;
            A2(n,n) = A2(n,n) + eps;
        end
        % solve
        x1 = A1 \ b1;
        x2 = A2 \ b2;
        % relative error
        x = ones(n, 1);
        err1 = norm(x1 - x) / norm(x);
        err2 = norm(x2 - x) / norm(x);
        if perturb_b
            err_lim1 = cond(org_A1) * norm(b1 - org_b1) / norm(org_b1);
            err_lim2 = cond(org_A2) * norm(b2 - org_b2) / norm(org_b2);
        else
            err_A1 = norm(A1 - org_A1) / norm(A1);
            err_lim1 = cond(org_A1) / (1 - cond(org_A1) * err_A1) * err_A1;
            err_A2 = norm(A2 - org_A2) / norm(A2);
            err_lim2 = cond(org_A2) / (1 - cond(org_A2) * err_A2) * err_A2;
        end
        % output
        fprintf('n: %d, cond(A1): %g, cond(A2): %g\n', n, cond(A1), cond(A2));
        fprintf('n: %d, eps: %g, err1: %.4f/%.4f, err2: %.4f/%.4f\n', ...
            n, eps, err1, err_lim1, err2, err_lim2);
        x1'
        x2'
    end
end