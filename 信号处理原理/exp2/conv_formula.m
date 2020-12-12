function result = conv_formula( x, y )
    % CONV_FORMULA calculate linear convolution based on formula
    
    len = length(x) + length(y) - 1;
    result = zeros(1, len);

    % the padded sequence should look like this
    % 0, ..., 0, a_1, ...,  a_n
    % b_m, ...,  b_1, 0, ..., 0
    pad_a = [zeros(1, length(y) - 1), x];
    pad_b = [fliplr(y), zeros(1, length(x) - 1)];

    % formula
    for i = 1:len
        result(i) = sum(pad_a .* pad_b);
        pad_b = [0, pad_b(1:len - 1)];
    end
end
