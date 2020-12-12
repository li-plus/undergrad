x = 60 * [0 2 4 5 6 7 8 9 10.5 11.5 12.5 14 16 17 18 19 20 21 22 23 24];
y = [2 2 0 2 5 8 25 12 5 10 12 7 9 28 22 10 9 11 8 9 3];

xq = 0:24 * 60;
yq = interp1(x, y, xq, 'spline');
yq(yq < 0) = 0;

figure;
plot(xq, yq);
hold on;
scatter(x, y);
xlim([0, 24 * 60]);
ylim([0, 30]);
xlabel('Time (min)');
ylabel('Traffic Flow');
day_flow = trapz(xq, yq);
fprintf('Total traffic flow per day: %.0f\n', day_flow);
