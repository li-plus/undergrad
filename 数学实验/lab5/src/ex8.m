t = [0.083 0.167 0.25 0.50 0.75 1.0 1.5 2.25 3.0 4.0 6.0 8.0 10.0 12.0]';
c = [10.9 21.1 27.3 36.4 35.5 38.4 34.8 24.2 23.6 15.7 8.2 8.3 2.2 1.8]';

x0 = [0.1 1 10]';

% options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
%     'HessUpdate', 'bfgs', 'MaxFunEvals', 1000000, 'MaxIter', 10000);
% [x, fval, exitflag, output] = fminunc(@fun, x0, options, t, c);

options = optimoptions('lsqcurvefit', 'Algorithm', 'levenberg-marquardt',...
    'MaxFunEvals', 1000000, 'MaxIter', 10000);
[x,resnorm,residual,exitflag,output] = ...
    lsqcurvefit(@lsqfun, x0, t, c, [], [], options);

k = x(1); k1 = x(2); b = x(3);
t_plot = 0:0.1:12;

figure; plot(t_plot, concentration(k, k1, b, t_plot));
hold on; scatter(t, c); xlabel('t'); ylabel('c(t)');

function c = concentration(k, k1, b, t)
    c = b * k1 / (k1 - k) * (exp(-k*t) - exp(-k1*t));
end

function err = fun(x, t, c)
    k = x(1); k1 = x(2); b = x(3);
    err = sum((concentration(k, k1, b, t) - c).^2);
end

function c = lsqfun(x, t)
    k = x(1); k1 = x(2); b = x(3);
    c = concentration(k, k1, b, t);
end
