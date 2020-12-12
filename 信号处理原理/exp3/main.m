% load audio data
[audio1, fs] = audioread('data/audio1.wav');
fprintf('load audio at %dHz\n', fs);
[audio2, fs] = audioread('data/audio2.wav');
fprintf('load audio at %dHz\n', fs);
[audio3, fs] = audioread('data/audio3.wav');
fprintf('load audio at %dHz\n', fs);

% crop audio to the same size (seq_len)
seq_len = 33000;
audio1 = audio1(5200 : 5200 + seq_len - 1);
audio2 = audio2(42000 : 42000 + seq_len - 1);
audio3 = audio3(length(audio3) - seq_len + 1 : length(audio3));

seqs = [audio1, audio2, audio3];
num_seqs = size(seqs, 2);

% play original wav
for i = 1:num_seqs
    fprintf('playing audio %d\n', i);
    soundsc(seqs(:, i), fs);
    pause(seq_len / fs);
end

% plot original signal on time and frequency domain
figure(1);
for i = 1:num_seqs
    subplot(num_seqs, 2, 2 * i - 1);
    plot(1:seq_len, seqs(:, i));
    subplot(num_seqs, 2, 2 * i);
    plot(1:seq_len, abs(fft(seqs(:, i))));
end

% -------------- start encoding --------------

% upsample every sequence by num_seqs times
seqs = upsample(seqs, num_seqs);

% plot signals on time and frequency domain after upsampling
figure(2);
for i = 1:num_seqs
    subplot(num_seqs, 2, 2 * i - 1);
    plot(1:seq_len * num_seqs, seqs(:, i));
    subplot(num_seqs, 2, 2 * i);
    plot(1:seq_len * num_seqs, abs(fft(seqs(:, i))));
end

% get frequency
for i = 1:num_seqs
    seqs(:, i) = fft(seqs(:, i));
end

% filter and sum to get modulated data
merged_freq = zeros(num_seqs * seq_len, 1);
band = seq_len / 2;
filter_lo = [1, band];
filter_hi = [num_seqs * seq_len - band + 1, num_seqs * seq_len];
for i = 1:num_seqs
    merged_freq(filter_lo(1):filter_lo(2)) = seqs(filter_lo(1):filter_lo(2), i);
    merged_freq(filter_hi(1):filter_hi(2)) = seqs(filter_hi(1):filter_hi(2), i);
    % update filters to avoid overlapping
    filter_lo = filter_lo + band;
    filter_hi = filter_hi - band;
end

% use ifft to get modulated data, which is to be transitted. 
% it has a small imaginary part due to matlab error,
% so the real part is explicitly extracted.
merged = real(ifft(merged_freq));

% plot transimitted signal
figure(3);
subplot(211);
plot(1:length(merged), merged);
subplot(212);
plot(1:length(merged_freq), abs(merged_freq));

% -------------- start decoding --------------

% get frequency
merged_freq = fft(merged);

% filter to get every sequence
seqs = zeros(seq_len, num_seqs);
band = seq_len / 2;
filter_lo = [1, band];
filter_hi = [num_seqs * seq_len - band + 1, num_seqs * seq_len];
for i = 1:num_seqs
    % order is important here
    if mod(i, 2)
        seqs(:, i) = [merged_freq(filter_lo(1):filter_lo(2)); merged_freq(filter_hi(1):filter_hi(2))];
    else
        seqs(:, i) = [merged_freq(filter_hi(1):filter_hi(2)); merged_freq(filter_lo(1):filter_lo(2))];
    end
    filter_lo = filter_lo + band;
    filter_hi = filter_hi - band;
end

% get time domain signal
for i = 1:num_seqs
    seqs(:, i) = real(ifft(seqs(:, i)));
end

% plot decoded data
figure(4);
for i = 1:num_seqs
    subplot(num_seqs, 2, 2 * i - 1);
    plot(1:seq_len, seqs(:, i));
    subplot(num_seqs, 2, 2 * i);
    plot(1:seq_len, abs(fft(seqs(:, i))));
end

% play decoded audio
for i = 1:num_seqs
    fprintf('playing audio %d\n', i);
    soundsc(seqs(:, i), fs);
    pause(seq_len / fs);
end
