clear; clc;

%% =============== PARAMETERS ===============
ris_port = 'COM22';
baudrate = 115200;
period = 0.5e-3 / 200;  % seconds
duration_per_pattern = 3;  % seconds
sleep_time = 0;
file_max = 'optimized_max_ris_pattern_hex.txt';
file_min = 'optimized_min_ris_pattern_hex.txt';

mseq = [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0];

%% =============== INIT SERIAL (RIS) ===============
ris = serialport(ris_port, baudrate);
writeline(ris, '!Reset');
pause(1);
while ris.NumBytesAvailable > 0
    response = readline(ris);
    fprintf("Reset response: %s\n", response);
    pause(0.1);
end

%% =============== PARSE HEX FILES ===============
max_patterns = read_patterns(file_max);
min_patterns = read_patterns(file_min);
num_patterns = min(length(max_patterns), length(min_patterns));

%% =============== MAIN LOOP ===============
disp('Press Ctrl+C to stop.')

try
    while true
        for i = 1:num_patterns
            high = max_patterns(i).hex;
            low  = min_patterns(i).hex;
            idx1 = max_patterns(i).index;
            idx2 = min_patterns(i).index;

            fprintf('\nRunning sequence for pair %d:\n', i);
            fprintf('→ Max pattern [%d,%d]: %s\n', idx1(1), idx1(2), high);
            fprintf('→ Min pattern [%d,%d]: %s\n', idx2(1), idx2(2), low);

            ris_sequence(ris, high, low, mseq, period, duration_per_pattern, sleep_time);
        end
    end
catch
    disp("Process interrupted by user.");
end

fclose(ris);

%% ================= FUNCTIONS =================

function patterns = read_patterns(filename)
    fid = fopen(filename, 'r');
    patterns = struct('hex', {}, 'index', {});
    line = fgetl(fid);
    while ischar(line)
        expr = '!0x[\da-fA-F]+';
        hex = regexp(line, expr, 'match', 'once');
        index = regexp(line, '\[(\d+),\s*(\d+)\]', 'tokens', 'once');
        if ~isempty(hex) && ~isempty(index)
            patterns(end+1).hex = hex; %#ok<*AGROW>
            patterns(end).index = [str2double(index{1}), str2double(index{2})];
        end
        line = fgetl(fid);
    end
    fclose(fid);
end

function ris_sequence(ris, high, low, mseq, period, duration, sleep_time)
    t_start = tic;
    while toc(t_start) < duration
        for bit = mseq
            if bit == 1
                send_pattern(ris, high);
            else
                send_pattern(ris, low);
            end
            pause(period);
        end
        pause(sleep_time);
    end
end

function send_pattern(ris, pattern)
    writeline(ris, pattern);
    pause(0.05);
    while ris.NumBytesAvailable > 0
        response = readline(ris);
        fprintf("Pattern response: %s\n", response);
        pause(0.05);
    end
end
