function [row_index, col_index] = dtmf_fft(seq, fs, dtmf_row_freqs, dtmf_col_freqs)
    % DTMF_FFT recognize dtmf signals using fft algorithm
    
    % note that N_min = f_s / delta_f. so just let N = f_s.
    amplitude = abs(fft(seq, fs));
    
    % find row frequency from [624, 1030]
    [~, row_freq] = max(amplitude(624:1030));
    % add array offset and minus one, because matlab index starts from 1
    row_freq = 623 + row_freq - 1;
    
    % find column frequency from [1082, 1789]
    [~, col_freq] = max(amplitude(1082:1789));
    % add array offset and minus one, because matlab index starts from 1
    col_freq = 1081 + col_freq - 1;
    
    % find the row and column indices of the closest frequency
    [~, row_index] = min(abs(dtmf_row_freqs - row_freq));
    [~, col_index] = min(abs(dtmf_col_freqs - col_freq));
end
