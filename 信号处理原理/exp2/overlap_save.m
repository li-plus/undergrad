function result = overlap_save( x, y )
    % OVERLAP_SAVE calculate linear convolution using overlap-save algorithm.
    
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
    
    % init overlap by zeros
	overlap = zeros(1, len_y - 1);
    
    % init result with a larger space to avoid index out of range
	result = zeros(1, (ceil(len_x / block_size) + 1) * block_size);
    
    % pad x to a multiple of block_size to make our lives easier
	pad_x = [x, zeros(1, length(result) - len_x)];
    
    % for each block in x, concatenate it with overlap part, which is 
    % updated regularly, and then compute its convolution result with y. 
    % the last block_size points in every block result are valid.
    % concatenate them and finally we get the results.
	for lo = 1:block_size:length(pad_x)
		block = [overlap, pad_x(lo:lo + block_size - 1)];
		overlap = block(block_size + 1 : block_size + len_y - 1);
		block_result = ifft(fft(block, N) .* fft_y);
		result(lo:lo + block_size - 1) = block_result(len_y:len_y + block_size - 1);
    end
    
    % only return the first (len_x + len_y - 1) elements
	result = result(1:len_x + len_y - 1);
end
