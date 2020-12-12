global d;
global v1;
global v2;

d = 100;
v2 = 2;
y = 0:0.1:d;
v1_choices = [1 2 4];

gt_x = @(y, d, v1, v2) (d-y)./2 .* ((d./(d-y)).^(v1 / v2) - (d./(d-y)).^(-v1 / v2));

figure;
for i = 1:3
    subplot(1, 3, i);
    v1 = v1_choices(i);
    x = gt_x(y, d, v1, v2);
    plot(x, y);
    xlabel('x');
    ylabel('y');
    if (max(x) > 2000)
        xlim([0 2000]);
    end
    title(['v_1 / v_2 = ' num2str(v1/v2)]);
end

v1_choices = [0.5 1.5 2];

v1 = 2;
[t, xy] = ode23s(@dxy_dt, 0:200, [0 0]);
x = xy(:, 1);
y = xy(:, 2);
figure;
yyaxis left;
plot(t, x);
ylabel('x (m)');
yyaxis right;
plot(t, y);
ylabel('y (m)');
ylim([0 100]);
xlabel('t (s)');
legend('x(t)', 'y(t)', 'Location', 'northwest');
figure;
plot(x, y, 'r--', 'lineWidth', 2);
ylim([0 100]);
hold on;
gt_y = 0:0.01:100;
plot(gt_x(gt_y, d, v1, v2), gt_y, 'b');
xlabel('x (m)');
ylabel('y (m)');
legend('Numerical', 'Analytical');

function result = dxy_dt(t, xy)
global d;
global v1;
global v2;
x = xy(1);
y = xy(2);
len = sqrt(x.^2 + (d-y).^2);
cos_theta = (d-y) ./ len;
sin_theta = x ./ len;
result = [v1 - v2 * sin_theta; v2 * cos_theta];
end
