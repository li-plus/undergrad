% max total length L for testing
max_len = 5000;

% init time consuming results for 4 algorithms
time_conv_formula = zeros(1, max_len);
time_conv_fft = zeros(1, max_len);
time_overlap_add = zeros(1, max_len);
time_overlap_save = zeros(1, max_len);

for i = 2:max_len
    % randomly choose L_x from [1, L - 1]
    len_x = randi(i - 1);
    % let L_y = L - L_x, so that L_x + L_y = L
    len_y = i - len_x;

    % randomly init sequence x and y
    x = randi(100, 1, len_x);
    y = randi(100, 1, len_y);

    % test 4 algorithms separately
    tic
    ans1 = conv_formula(x, y);
    time_conv_formula(i) = toc;
    
    tic
    ans2 = conv_fft(x, y);
    time_conv_fft(i) = toc;
    
    tic
    ans3 = overlap_add(x, y);
    time_overlap_add(i) = toc;
    
    tic
    ans4 = overlap_save(x, y);
    time_overlap_save(i) = toc;

    % regularly logging
    if mod(i, 500) == 0
        fprintf('%d done\n', i);
    end

    % check correctness
    isclose = @(x, y) abs(x - y) < 1e-6;
    if(~(all(isclose(ans1, ans2)) && all(isclose(ans2, ans3)) && all(isclose(ans3, ans4))))
        fprintf('Wrong!\n');
    end
end

% plot results
temporal = 1:max_len;
plot(temporal, [time_conv_formula;time_conv_fft;time_overlap_add;time_overlap_save]);
xlabel('Total Length');
ylabel('Time(s)');
legend({'Formula', 'FFT', 'Overlap-Add', 'Overlap-Save'}, 'Location', 'northwest');
