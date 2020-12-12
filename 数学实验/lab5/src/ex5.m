dist = [
    4 1 0.9607; 5 4 0.4758; 18 8 0.8363; 15 13 0.5725;
    12 1 0.4399; 12 4 1.3402; 13 9 0.3208; 19 13 0.7660;
    13 1 0.8143; 24 4 0.7006; 15 9 0.1574; 15 14 0.4394;
    17 1 1.3765; 8 6 0.4945; 22 9 1.2736; 16 14 1.0952;
    21 1 1.2722; 13 6 1.0559; 11 10 0.5781; 20 16 1.0422;
    5 2 0.5294; 19 6 0.6810; 13 10 0.9254; 23 16 1.8255;
    16 2 0.6144; 25 6 0.3587; 19 10 0.6401; 18 17 1.4325;
    17 2 0.3766; 8 7 0.3351; 20 10 0.2467; 19 17 1.0851;
    25 2 0.6893; 14 7 0.2878; 22 10 0.4727; 20 19 0.4995;
    5 3 0.9488; 16 7 1.1346; 18 11 1.3840; 23 19 1.2277;
    20 3 0.8000; 20 7 0.3870; 25 11 0.4366; 24 19 1.1271;
    21 3 1.1090; 21 7 0.7511; 15 12 1.0307; 23 21 0.7060;
    24 3 1.1432; 14 8 0.4439; 17 12 1.3904; 23 22 0.8052;
]';

coords0 = zeros(48,1);

% options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
%     'HessUpdate', 'bfgs', 'MaxFunEvals', 1000000, 'MaxIter', 10000);
% [coords, fval, exitflag, output] = fminunc(@fun, coords0, options, dist);

options = optimoptions('lsqnonlin', 'Algorithm', 'levenberg-marquardt',...
    'MaxFunEvals', 1000000, 'MaxIter', 10000);
[coords,resnorm,residual,exitflag,output] = ...
    lsqnonlin(@lsqfun, coords0, [], [], options, dist);

x = [coords(1:2:48); 0];
y = [coords(2:2:48); 0];
figure; scatter(x, y); xlabel('x'); ylabel('y'); grid on;
text(x, y+0.1, num2str((1:length(x))'), 'HorizontalAlignment', 'center');

function [x, y] = get_coord(coords, index)
    if index == length(coords) / 2 + 1
        x = 0;
        y = 0;
    else
        x = coords(2 * index - 1);
        y = coords(2 * index);
    end
end

function err = fun(coords, dist)
    err = 0;
    for pair = dist
        d_ij = pair(3);
        [x_i, y_i] = get_coord(coords, pair(1));
        [x_j, y_j] = get_coord(coords, pair(2));
        err = err + (sqrt((x_i - x_j)^2 + (y_i - y_j)^2) - d_ij)^2;
    end
end

function y = lsqfun(coords, dist)
    y = zeros(size(dist, 2), 1);
    for i = 1:length(y)
        pair = dist(:, i);
        d_ij = pair(3);
        [x_i, y_i] = get_coord(coords, pair(1));
        [x_j, y_j] = get_coord(coords, pair(2));
        y(i) = sqrt((x_i - x_j)^2 + (y_i - y_j)^2) - d_ij;
    end
end
