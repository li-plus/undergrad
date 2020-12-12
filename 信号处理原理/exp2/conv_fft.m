function result = conv_fft( x, y )
    % CONV_FFT calculate linear convolution using N-point FFT
    % if N >= len_x + len_y - 1, then circular_conv(x, y) == linear_conv(x, y)
    N = length(x) + length(y) - 1;
    result = ifft(fft(x, N) .* fft(y, N), N);
end
