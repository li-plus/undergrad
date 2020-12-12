data = [
    1 44.6 44 89.5 6.82 62 178
    2 45.3 40 75.1 6.04 62 185
    3 54.3 44 85.8 5.19 45 156
    4 59.6 42 68.2 4.90 40 166
    5 49.9 38 89.0 5.53 55 178
    6 44.8 47 77.5 6.98 58 176
    7 45.7 40 76.0 7.17 70 176
    8 49.1 43 81.2 6.51 64 162
    9 39.4 44 81.4 7.85 63 174
    10 60.1 38 81.9 5.18 48 170
    11 50.5 44 73.0 6.08 45 168
    12 37.4 45 87.7 8.42 56 186
    13 44.8 45 66.5 6.67 51 176
    14 47.2 47 79.2 6.36 47 162
    15 51.9 54 83.1 6.20 50 166
    16 49.2 49 81.4 5.37 44 180
    17 40.9 51 69.6 6.57 57 168
    18 46.7 51 77.9 6.00 48 162
    19 46.8 48 91.6 6.15 48 162
    20 50.4 47 73.4 6.05 67 168
    21 39.4 57 73.4 7.58 58 174
    22 46.1 54 79.4 6.70 62 156
    23 45.4 52 76.3 5.78 48 164
    24 54.7 50 70.9 5.35 48 146
];

y = data(:,2);
xs = data(:,3:7);
x1 = xs(:,1); x2 = xs(:,2); x3 = xs(:,3); x4 = xs(:,4); x5 = xs(:,5);

% plot scatter
if 0
    for i = 1:5
        figure; scatter(xs(:,i),y,'+'); grid on;
        xlabel(['x' int2str(i)]); ylabel('y');
    end
end

% single param
if 0
    X = [ones(size(y)) x3];
    [b,bint,r,rint,stats] = regress(y,X);
    figure; rcoplot(r,rint); xlabel('Case Number'); ylabel('Residuals'); title('');

    x = [4 9];
    yHat = b(1) + b(2) * x;
    figure; plot(x,yHat); hold on; scatter(x3,y,'+'); grid on; xlabel('x'); ylabel('y');

    polytool(x3,y);
end

% two param
if 0
    stepwise(xs, y);
    X = [ones(size(y)) x1 x3];
    [b,bint,r,rint,stats] = regress(y,X);
    figure; rcoplot(r,rint); xlabel('Case Number'); ylabel('Residuals'); title('');
end

% multi param
if 0
    stepwise(xs, y);
    X = [ones(size(y)) x1 x3 x5];
    [b,bint,r,rint,stats] = regress(y,X);
end

% outlier
if 1
    X = [ones(size(y)) x1 x3 x5];
    mask = true(size(y));
    mask(4) = 0; mask(10) = 0; mask(15) = 0; mask(17) = 0; mask(23) = 0;
    X = X(mask,:);
    y = y(mask);
    [b,bint,r,rint,stats] = regress(y,X);
    figure; rcoplot(r,rint); xlabel('Case Number'); ylabel('Residuals'); title('');
end
