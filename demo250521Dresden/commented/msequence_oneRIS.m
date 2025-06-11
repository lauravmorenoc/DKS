clear all

%% === Configuration ===

% Predefined test sequences for different RIS setups

% -- Sequences for 2-RIS testing --
mseq1 = [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0];   % For RIS 1
mseq2 = [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1];   % For RIS 2

% -- Sequences for 3-RIS testing --
mseq3_1 = [0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0]; % For RIS 1
mseq3_2 = [1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1]; % For RIS 2
mseq3_3 = [0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,0]; % For RIS 3

% >>> MODIFY THIS LINE TO SELECT THE SEQUENCE <<<
% Example options:
% mseq = mseq1;
% mseq = mseq2;
% mseq = mseq3_1;
% mseq = mseq3_2;
% mseq = mseq3_3;
mseq = mseq1;  % <-- SELECT THE SEQUENCE FOR THE CURRENT TEST

% >>> MODIFY THIS LINE TO SET THE CORRECT COM PORT <<<
ris = ris_init('COM6', 115200);   % <-- Set correct COM port for the target RIS

% Control signal format (RIS ON/OFF patterns)
all_off = '!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on  = '!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
high = all_off;  % Define logical "1"
low  = all_on;   % Define logical "0"

% Timing
period = 0.5e-3 / 200;  % Symbol duration
duration = 50;          % Total time to repeat the sequence
sleep_time = 0;         % Pause between full loops (not used here)

% Start RIS control
ris_sequence(ris, high, low, mseq, period, duration, sleep_time);

%% === Function Definitions ===

function ris = ris_init(port, baud)
    % Initialize the RIS over serial connection
    ris = serialport(port, baud);

    % Reset command
    writeline(ris, '!Reset');
    pause(1);

    % Read responses during reset
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

function ris_sequence(ris, high, low, sequence, period, duration, sleep_time)
    % Transmit binary sequence to RIS in loop
    time = 0;
    while time < duration
        for i = 1:length(sequence)
            if sequence(i) == 0
                currentPattern = low;
            elseif sequence(i) == 1
                currentPattern = high;
            else
                disp('Sequence must contain only 0 or 1');
                continue;
            end

            writeline(ris, currentPattern);
            readline(ris);  % Optional: Read response
            pause(period);
            time = time + period;
        end
        pause(sleep_time);
    end
end
