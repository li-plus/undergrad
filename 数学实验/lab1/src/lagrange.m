function vq = lagrange(x, y, xq)
    if (length(x) ~= length(y))
        error('Fixed points x and y must be of the same size');
    end

    vq = zeros(1, length(xq));
    for i = 1:length(x)
        p = ones(1, length(xq));
        for j = 1:length(x)
            if (i ~= j)
                p = p .* (xq - x(j)) / (x(i) - x(j));
            end
        end
        vq = vq + p .* y(i);
    end
end
