clear all

%% Variables

file_max = 'optimized_max_ris_pattern_hex.txt';
file_min = 'optimized_min_ris_pattern_hex.txt';

% Open files
fid_max = fopen(file_max, 'r');
fid_min = fopen(file_min, 'r');

% Read first line
max_pow_pattern = fgetl(fid_max);
min_pow_pattern = fgetl(fid_min);

% Close files
fclose(fid_max);
fclose(fid_min);

% Display result
disp(['Max power pattern: ', max_pow_pattern])
disp(['Min power pattern: ', min_pow_pattern])

all_off='!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on='!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
high=all_off;
low=all_on;
period=0.5e-3/200; % seconds
duration=50; % seconds
sleep_time=0;

% Sequences for 2 RIS testing
mseq1=[0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0];
mseq2=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1];

% Sequences for 3 RIS testing 
mseq3_1=[0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0];
mseq3_2=[1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1];
mseq3_3=[0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,0];

mseq=mseq3_1;
ris=ris_init('COM22', 115200);   % initialize RIS

%ris_sequence(ris, high, low, mseq, period, duration, sleep_time)
ris_sequence(ris, max_pow_pattern, min_pow_pattern, mseq, period, duration, sleep_time)




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


function ris_sequence(ris, high, low, sequence, period, duration, sleep_time)
    time=0;
    while (time<duration)
        %if (relative_time<sleep_time)
            for i=1:length(sequence)
                switch sequence(i)
                    case 0
                        currentPattern=low;
                    case 1
                        currentPattern=high;
                    otherwise
                        disp('Could not write sequence value, it must be either 0 or 1')
                end
                writeline(ris, currentPattern);
                % Get response
                response = readline(ris);
                %fprintf("Response from setting a pattern: %s\n", response);
                %fprintf("Current pattern: %s\n", currentPattern);
                %fprintf("Current symbol: %s\n", sequence(i));
                pause(period);
                time=time+period;
            end
            pause(sleep_time)
    end
end