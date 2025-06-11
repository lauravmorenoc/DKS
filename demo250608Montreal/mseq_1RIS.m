
clear all; clc
global stopFlag;
stopFlag = false;                    % Always close the file


%% Variables

all_off='!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on='!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
high=all_off;
low=all_on;
period=0.5e-3/200; % seconds
duration=50; % seconds
sleep_time=0;
RIS_index=0;
OFF=0;
ON=1;

% Saying the RIS is initially off
fid = fopen('output.txt', 'w');    % Open file for writing (overwrite mode)
fprintf(fid, '%d%d\n', RIS_index, OFF); 

% Sequences for 2 RIS testing
mseq1=[0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0];
mseq2=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1];

% Sequences for 3 RIS testing 
mseq3_1=[0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0];
mseq3_2=[1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1];
mseq3_3=[0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,0];

mseq=mseq1;
ris=ris_init('COM5', 115200);   % initialize RIS
f = figure('Name', 'Press Q to stop', 'KeyPressFcn', @(src, event) stop_on_q(event)); % Interruption
ris_sequence(ris, high, low, mseq, period, duration, sleep_time, RIS_index, ON, OFF)




%% Functions

function ris = ris_init(port, baud)
    % RIS initialization
    % Get a new RIS object from serial port
    ris = serialport(port, baud);
    
    % Reset RIS
    writeline(ris, '!Reset');
    % Wait long enough or check ris.NumBytesAvailable for becoming non-zero
    pause(1);
    while ris.NumBytesAvailable > 0
        response = readline(ris);
        fprintf("Response from resetting RIS: %s\n", response);
        pause(0.1);
    end
    
    % Clear input buffer
    pause(0.1);
    while ris.NumBytesAvailable > 0
        readline(ris);
        pause(0.1);
    end
end

function ris_sequence(ris, high, low, sequence, period, duration, sleep_time, RIS_index, ON, OFF)
    global stopFlag;
    time = 0;
    fid = fopen('output.txt', 'w');    % Open file for writing (overwrite mode)
    fprintf(fid, '%d%d\n', RIS_index, ON); 
    fclose(fid);

    while time < duration && ~stopFlag
        for i = 1:length(sequence)
            if stopFlag
                break;
            end

            if sequence(i) == 0
                currentPattern = low;
            elseif sequence(i) == 1
                currentPattern = high;
            else
                disp('Invalid sequence value');
                continue;
            end

            writeline(ris, currentPattern);
            readline(ris);  % Read response (optional)
            pause(period);
            time = time + period;
        end
        pause(sleep_time);
    end

    disp("Sequence completed or stopped by user.");
    fid = fopen('output.txt', 'w');    % Open file for writing (overwrite mode)
    fprintf(fid, '%d%d\n', RIS_index, OFF); 
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

