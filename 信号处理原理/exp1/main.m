% single key test
for i = '0':'9'
    % get audio sequence and sampling frequency fs
    audio_filename = ['data/dtmf-', i, '.wav'];
    [seq, fs] = audioread(audio_filename);
    fprintf(['Loaded ', audio_filename, ' with length ', int2str(length(seq)), '\n']);
    
    % dtmf definition
    dtmf_keys = ['1', '2', '3', 'A';
                 '4', '5', '6', 'B';
                 '7', '8', '9', 'C';
                 '*', '0', '#', 'D'];
    dtmf_row_freqs = [697, 770, 852, 941];
    dtmf_col_freqs = [1209, 1336, 1477, 1633];

    % solve with goertzel
    tic
    [row_index, col_index] = dtmf_goertzel(seq, fs, dtmf_row_freqs, dtmf_col_freqs);
    fprintf('Goertzel finished in %f s with key %s\n', toc, dtmf_keys(row_index, col_index));
    
    % solve with fft
    tic
    [row_index, col_index] = dtmf_fft(seq, fs, dtmf_row_freqs, dtmf_col_freqs);
    fprintf('FFT finished in %f s with key %s\n', toc, dtmf_keys(row_index, col_index)); 
end

% multiple key test
[multi_seq, fs] = audioread('data/13380831033.wav');

% plot signals on time domain
plot(1:350000, multi_seq(1:350000));

% init results
fft_result = '';
goertzel_result = '';

% starting point and end point of the audio and the interval between two keys
first = 57000;
step = 22000;
last = 277000;
key_length = 5000;

fft_time = 0;
goertzel_time = 0;

for lo = first:step:last
    seq = multi_seq(lo:lo + key_length - 1, 1);

    % goertzel
    tic
    [row_index, col_index] = dtmf_goertzel(seq, fs, dtmf_row_freqs, dtmf_col_freqs);
    goertzel_time = goertzel_time + toc;
    goertzel_result = strcat(goertzel_result, dtmf_keys(row_index, col_index));

    % fft
    tic
    [row_index, col_index] = dtmf_fft(seq, fs, dtmf_row_freqs, dtmf_col_freqs);
    fft_time = fft_time + toc;
    fft_result = strcat(fft_result, dtmf_keys(row_index, col_index));
end

fprintf('Goertzel finished in %f s with key %s\n', goertzel_time, goertzel_result);
fprintf('FFT finished in %f s with key %s\n', fft_time, fft_result);
