mu = 4.8;
d = 0.25;
r = 0.3;
fun = @(q, c) r/d*(c - atan(mu * q)) + (1-r) * q;

ps_fork1 = 1.02:0.001:1.14;
ps_fork2 = 0.9:0.001:1;
ps_fork3 = 0.86:0.001:0.94;
ps_fork4 = 0.8964:0.00001:0.8976;
ps_forks = 0.86:0.0001:1.09;
ps_global = -2:0.01:2;

ps = ps_forks;
max_iter = 1000;
x1 = 0.5;

q = chaos(fun, x1, ps, max_iter);
% figure; plot(0:150, q(10000, 1:151)); xlabel('t'); ylabel('q_t');
forks_index = get_forks(q);
forks = ps(forks_index);
forks
e = length(forks);
fegenbaum = (forks(2:e-1) - forks(3:e)) ./ (forks(1:e-2) - forks(2:e-1));
fegenbaum

figure; plot(ps, q(:, max_iter - 98:max_iter + 1), 'k.');
xlabel('c'); ylabel('q_n (n\rightarrow +\infty)');

function x = chaos(fun, x1, ps, max_iter)
% observe chaos while changing the param of the difference equation
% fun: iterated function of difference quation x_{n+1} = fun(x_n, p),
%       where the changeable param p might cause chaos
% x1: initial value of the sequence x, i.e., x_1
% ps: values of the changing param p to investigate
% max_iter: max iterations of the difference equation
% return: the sequence x under each param value specified in ps,
%       where x(i, j) is the value of x(j) under i-th param value
    x = zeros(length(ps), max_iter + 1);
    for p_index = 1:length(ps)
        p = ps(p_index);
        x(p_index, 1) = x1;
        for iter = 1:max_iter
            x(p_index, iter + 1) = fun(x(p_index, iter), p);
        end
    end
end

function forks_index = get_forks(xs)
    [num_p, max_iter] = size(xs);
    ways = ones(num_p, 1);
    for index = 1:num_p
        x = xs(index, :);
        while 1
            if max_iter < 2 * ways(index)
                ways(index) = -1;
                break;
            end
            last = x(max_iter - ways(index) + 1:max_iter);
            prev = x(max_iter - 2 * ways(index) + 1:max_iter - ways(index));
            if mean(abs(last - prev)) < 1e-6 * ways(index)
                break
            end
            ways(index) = ways(index) * 2;
        end
    end
    
    edge_cnt = 0;
    for i = 2:length(ways)
        if ways(i) ~= ways(i-1) && ways(i) ~= -1 && ways(i-1) ~= -1
            edge_cnt = edge_cnt + 1;
            forks_index(edge_cnt) = i;
        end
    end
end
