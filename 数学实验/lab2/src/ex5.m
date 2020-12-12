G = 527.436;
g = 32.17405;
m = G / g;
k = 0.08;
v0 = 40;
F = 470.327;
dv_dt = @(t, v) (G - F - k * v) / m;

[t, v] = ode45(dv_dt, 0:0.1:14, 0);

s = zeros(length(t), 1);
for i = 2:length(t)
    s(i) = trapz(t(1:i), v(1:i));
end

figure;
plot(t, v);
grid on;
xlabel('Time (s)');
ylabel('Velocity (ft/s)');

figure;
plot(t, s);
grid on;
xlabel('Time (s)');
ylabel('Depth (ft)');

figure;
gt_v = @(t) (G - F) / k * (1 - exp(-k / m * t));
plot(t, v - gt_v(t));
grid on;
xlabel('Time (s)');
ylabel('Residual of velocity (ft/s)');
