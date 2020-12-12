data = [
    1 6 28.1
    1 7 32.3
    1 8 34.8
    1 9 38.2
    1 10 43.5
    2 6 65.3
    2 7 67.7
    2 8 69.4
    2 9 72.2
    2 10 76.9
    3 6 82.2
    3 7 85.3
    3 8 88.1
    3 9 90.7
    3 10 93.6
];
x1 = data(:,1); x2 = data(:,2); y = data(:,3);

% x1 as a normal param
if 0
    X = [x1 x2];
    X1 = [ones(size(y)) X];
    stepwise(X, y);
    [b,bint,r,rint,stats] = regress(y,X1);
    figure; rcoplot(r,rint); xlabel('Case Number'); ylabel('Residuals'); title('');
end

% use two param
if 0
    x1b = double(x1 == 2);
    x1a = double(x1 == 3);
    X = [x1a x1b x2];
    X1 = [ones(size(y)) X];
    stepwise(X,y);
    mask = true(size(y));
    mask(5) = 0;
    [b,bint,r,rint,stats] = regress(y(mask),X1(mask,:));
    figure; rcoplot(r,rint); xlabel('Case Number'); ylabel('Residuals'); title('');
end

% add interaction
if 1
    X = [x1a x1b x2 x1a.*x2 x1b.*x2];
    X1 = [ones(size(y)) X];
    stepwise(X,y);
    [b,bint,r,rint,stats] = regress(y,X1);
    figure; rcoplot(r,rint); xlabel('Case Number'); ylabel('Residuals'); title('');
end
