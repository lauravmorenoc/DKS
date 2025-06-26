clear all; clc
global stopFlag;
stopFlag = false;                    % Always close the file


%% === Configuration ===

% These two lines must be updated with the correct COM ports before the demo
ris1 = ris_init('COM18', 115200);    % <-- Set the correct COM port for RIS 1
ris2 = ris_init('COM19', 115200);   % <-- Set the correct COM port for RIS 2

% Control signals
all_off = '!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on  = '!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
high = all_off;
low  = all_on;

% Timing parameters
period = 0.5e-3 / 200;    % Duration of each symbol
duration = 10;            % Total execution time in seconds
sleep_time = 0;           % Delay after each full sequence
time_in_between = period / 4;  % Delay between the two RIS commands
RIS_index=0;
OFF=0;
ON=1;

% Saying the RISs are initially off
fid = fopen('output.txt', 'w');    % Open file for writing (overwrite mode)
fprintf(fid, '%d%d\n', OFF, OFF); 

% Sequences for each RIS (must be the same length)
mseq1 = [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0];  % Sequence for RIS 1
mseq2 = [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1];  % Sequence for RIS 2

f = figure('Name', 'Press Q to stop', 'KeyPressFcn', @(src, event) stop_on_q(event)); % Interruption

% Start the dual sequence transmission
ris_two_seqs(ris1, ris2, high, low, mseq1, mseq2, period, duration, sleep_time, time_in_between, ON, OFF);



%% === Function Definitions ===

function ris = ris_init(port, baud)
    % Initialize RIS communication over serial
    ris = serialport(port, baud);
    
    % Reset the RIS
    writeline(ris, '!Reset');
    pause(1);
    while ris.NumBytesAvailable > 0
        response = readline(ris);
        fprintf("Response from resetting RIS: %s\n", response);
        pause(0.1);
    end
    
    % Clear any remaining data in the buffer
    pause(0.1);
    while ris.NumBytesAvailable > 0
        readline(ris);
        pause(0.1);
    end
end

function ris_two_seqs(ris1, ris2, high, low, sequence1, sequence2, period, duration, sleep_time, time_in_between, ON, OFF)
    global stopFlag;
    fid = fopen('output.txt', 'w');
    fprintf(fid, '%d%d\n', ON, ON); 
    fclose(fid);
    % Sends two sequences to two RISs in parallel, with timing control
    time = 0;
    while time < duration && ~stopFlag
        for i = 1:length(sequence1)
            % Select pattern for RIS 1
            if sequence1(i) == 0
                currentPattern1 = low;
            elseif sequence1(i) == 1
                currentPattern1 = high;
            else
                disp('Sequence 1: Invalid symbol (must be 0 or 1)');
                continue;
            end

            % Select pattern for RIS 2
            if sequence2(i) == 0
                currentPattern2 = low;
            elseif sequence2(i) == 1
                currentPattern2 = high;
            else
                disp('Sequence 2: Invalid symbol (must be 0 or 1)');
                continue;
            end

            % Send command to RIS 1
            writeline(ris1, currentPattern1);
            pause(time_in_between);

            % Send command to RIS 2
            writeline(ris2, currentPattern2);

            % Optional: Read response from both RISs (can be commented out)
            readline(ris1);
            readline(ris2);

            time = time + period;
            pause(period);
        end
        pause(sleep_time);
    end

    disp("Sequence completed or stopped by user.");
    fid = fopen('output.txt', 'w');    
    fprintf(fid, '%d%d\n', OFF, OFF); 
    fclose(fid);

end

function stop_on_q(event)
    global stopFlag;
    if strcmp(event.Key, 'q')
        stopFlag = true;
        disp('Stop requested by user.');
        close(gcf);  % Close the figure
    end
end

