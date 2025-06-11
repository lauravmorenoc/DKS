clear all

%% === Configuration ===

% Control patterns: 'all_off' = reflective, 'all_on' = transparent
all_off = '!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on  = '!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
high = all_off;
low  = all_on;

% Timing
period = 0.5e-3 / 200;      % Duration of each symbol
duration = 10;              % Total execution time (seconds)
sleep_time = 0;             % Optional delay after each sequence loop
time_in_between = period / 4;  % Delay between sequential commands to each RIS

% Sequences for 3-RIS testing (must all be same length)
mseq1 = [0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0];  % Sequence for RIS 1
mseq2 = [1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1];  % Sequence for RIS 2
mseq3 = [0,0,1,1,0,1,1,1,1,0,1,1,0,1,0,0];  % Sequence for RIS 3

% >>> MODIFY THESE LINES TO SET THE CORRECT COM PORTS <<<
ris1 = ris_init('COM6', 115200);   % <-- Replace with the actual COM port for RIS 1
ris2 = ris_init('COM7', 115200);   % <-- Replace with the actual COM port for RIS 2
ris3 = ris_init('COM8', 115200);   % <-- Replace with the actual COM port for RIS 3

% Start sequence transmission
ris_three_seqs(ris1, ris2, ris3, high, low, mseq1, mseq2, mseq3, period, duration, sleep_time, time_in_between);

%% === Function Definitions ===

function ris = ris_init(port, baud)
    % Initialize RIS over serial connection
    ris = serialport(port, baud);

    % Send reset command
    writeline(ris, '!Reset');
    pause(1);

    % Flush response from reset
    while ris.NumBytesAvailable > 0
        response = readline(ris);
        fprintf("Response from resetting RIS: %s\n", response);
        pause(0.1);
    end

    % Final cleanup
    pause(0.1);
    while ris.NumBytesAvailable > 0
        readline(ris);
        pause(0.1);
    end
end

function ris_three_seqs(ris1, ris2, ris3, high, low, seq1, seq2, seq3, period, duration, sleep_time, delay)
    % Send 3 sequences to 3 RISs in a coordinated loop
    time = 0;
    while time < duration
        for i = 1:length(seq1)
            % Translate bits to patterns
            currentPattern1 = conditional_pattern(seq1(i), high, low, "RIS1");
            currentPattern2 = conditional_pattern(seq2(i), high, low, "RIS2");
            currentPattern3 = conditional_pattern(seq3(i), high, low, "RIS3");

            % Send patterns with short delays between them
            writeline(ris1, currentPattern1);
            pause(delay);
            writeline(ris2, currentPattern2);
            pause(delay);
            writeline(ris3, currentPattern3);

            % Optional: read responses (can be commented out)
            readline(ris1);
            readline(ris2);
            readline(ris3);

            time = time + period;
            pause(period);
        end
        pause(sleep_time);
    end
end

function pattern = conditional_pattern(bit, high, low, ris_name)
    % Convert bit to pattern and check for validity
    if bit == 0
        pattern = low;
    elseif bit == 1
        pattern = high;
    else
        disp(['Invalid symbol in sequence for ', ris_name, '. Must be 0 or 1.']);
        pattern = low;  % Fallback
    end
end
