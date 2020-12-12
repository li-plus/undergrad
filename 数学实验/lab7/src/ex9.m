x1 = 0:500;
y1 = x1;
x2 = 501:1000;
y2 = 0.8*x2 + 100;
x3 = 1001:1500;
y3 = 0.6*x3 + 300;
plot(x1, y1, x2, y2, x3, y3);
xlabel('Purchase (t)'); ylabel('Total Cost (10k)'); grid on;
