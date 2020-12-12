function [row_index, col_index] = dtmf_goertzel(seq, fs, dtmf_row_freqs, dtmf_col_freqs) 
    % DTMF_GOERTZEL recognize dtmf signals using goertzel algorithm
    
    % get all dtmf frequencies
    dtmf_freqs = [dtmf_row_freqs, dtmf_col_freqs]';
    % get digital frequencies.
    w_k = dtmf_freqs * 2 * pi / fs;
    % precompute filters.
    cos_w_k_x2 = 2 * cos(w_k);

    % for N-point dft.
    N = length(seq);
    % init amplitude |X(k)|
    amplitude = zeros(1, length(dtmf_freqs));

    % for every dtmf frequency, compute v_k(n) and |X(k)|
    for k = 1:length(dtmf_freqs)
        prev = 0;
        curr = 0;
        for n = 3:N+2
            % recurrently compute v_k(n) using goertzel algorithm.
            next = seq(n-2) + cos_w_k_x2(k) * curr - prev;
            prev = curr;
            curr = next;
        end
        % since X(k) = y_k(N) = v(N) - v(N-1) * W_N^k. from Euler formula, 
        % we have |X(k)| = v^2(N) + v^2(N-1) - 2 * cos(w_k) * v(N) * v(N-1)
        amplitude(k) = curr^2 + prev^2 - cos_w_k_x2(k) * curr * prev;
    end

    % get row and column indices of the pressed key
    [~, row_index] = max(amplitude(1:4));
    [~, col_index] = max(amplitude(5:8));
end
