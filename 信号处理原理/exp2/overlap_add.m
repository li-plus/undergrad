function result = overlap_add( x, y )
    % OVERLAP_ADD calculate linear convolution using overlap-add algorithm.

    % ensure that x is longer than y
    if(length(x) < length(y))
        tmp = x;
        x = y;
        y = tmp;
    end
    
    len_x = length(x);
    len_y = length(y);
    
    block_size = len_y;
    
    % only calculate N-point fft(y) once
    N = block_size + len_y - 1;
    fft_y = fft(y, N);
    
    % init result with a larger space to avoid index out of range
    result = zeros(1, (1 + ceil(len_x / block_size)) * block_size);

    % for each block in x, compute its convolution with y.
    % sum them up, and then we get the result.
    for lo = 1:block_size:len_x
        hi = min(lo + block_size - 1, len_x);
        block_result = ifft(fft(x(lo:hi), N) .* fft_y, N);
        result(lo:lo + N - 1) = result(lo:lo + N - 1) + block_result;
    end
    % the first len_x + len_y - 1 points are valid results
    result = result(1:len_x + len_y - 1);
end
