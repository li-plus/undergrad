data = [
    1981 0 43.65
    1982 1 109.86
    1983 2 187.21
    1984 3 312.67
    1985 4 496.58
    1986 5 707.65
    1987 6 960.25
    1988 7 1238.75
    1989 8 1560.00
    1990 9 1824.29
    1991 10 2199.00
    1992 11 2438.89
    1993 12 2737.71
];

t = data(:,2);
yt = data(:,3);

% logistic linear
L = 3000;
y = log(L ./ yt - 1);
t1 = [ones(size(yt)) t];
[b,bint,r,rint,stats] = regress(y,t1);
a = exp(b(1));
k = -b(2);
beta = [L a k];
yt_hat = logistic(beta,t);
mse = sum((yt_hat - yt).^2) / (length(yt) - 2);

% logistic nonlinear
beta0 = [L a k];
[beta,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(t, yt, @logistic, beta0);

% gompertz nonlinear
L = 3000; b = 30; k = 0.4;
beta0 = [L b k];
[beta,R,J,CovB,MSE,ErrorModelInfo] = nlinfit(t, yt, @gompertz, beta0);

% plot
t_plt = 0:0.01:12;
yt_plt = gompertz(beta, t_plt);
figure; plot(t_plt, yt_plt); hold on; scatter(t, yt);
grid on; xlabel('t'); ylabel('y_t');

function yt = logistic(beta,t)
    L = beta(1); a = beta(2); k = beta(3);
    yt = L ./ (1 + a * exp(-k .* t));
end

function yt = gompertz(beta,t)
    L = beta(1); b = beta(2); k = beta(3);
    yt = L * exp(-b * exp(-k * t));
end
