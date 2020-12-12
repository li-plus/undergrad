global r1;
global r2;
global n1;
global n2;
global s1;
global s2;

r1=1;r2=2;n1=400;n2=300;s1=1.5;s2=1.7;x0=100;y0=100;

[t, xy] = ode45(@dxy_dt, 0:0.1:20, [x0 y0]);
x = xy(:, 1);
y = xy(:, 2);

figure;
plot(t, x);
hold on;
plot(t, y);
legend('x(t)', 'y(t)');
xlabel('t');
ylabel('x(t), y(t)');

figure;
plot(x, y)
xlabel('x');
ylabel('y');

function result = dxy_dt(t, xy)
    global r1;
    global r2;
    global n1;
    global n2;
    global s1;
    global s2;
    x = xy(1);
    y = xy(2);
    result = [r1*x*(1-x/n1-s1*y/n2); r2*y*(1-s2*x/n1-y/n2)];
end
