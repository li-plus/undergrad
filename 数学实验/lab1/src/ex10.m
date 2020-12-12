x = [0 3 5 7 9 11 12 13 14 15];
y1 = [0 1.8 2.2 2.7 3.0 3.1 2.9 2.5 2.0 1.6];
y2 = [0 1.2 1.7 2.0 2.1 2.0 1.8 1.2 1.0 1.6];

xq = 0:0.1:15;

y1q = lagrange(x, y1, xq);
y2q = lagrange(x, y2, xq);
plot_result(xq, y1q, y2q, x, y1, y2);
fprintf('Lagrange gives: %f\n', trapz(xq, y1q - y2q));

y1q = interp1(x, y1, xq, 'linear');
y2q = interp1(x, y2, xq, 'linear');
plot_result(xq, y1q, y2q, x, y1, y2);
fprintf('Linear gives: %f\n', trapz(xq, y1q - y2q));

y1q = interp1(x, y1, xq, 'spline');
y2q = interp1(x, y2, xq, 'spline');
plot_result(xq, y1q, y2q, x, y1, y2);
fprintf('Spline gives: %f\n', trapz(xq, y1q - y2q));

function plot_result(xq, y1q, y2q, x, y1, y2)   
    figure;
    scatter(x, y1, 36, 'r'); hold on;
    scatter(x, y2, 36, 'r'); hold on;
    plot(xq, y1q, 'b', xq, y2q, 'b');
    xlabel('x');
    ylabel('y');
end