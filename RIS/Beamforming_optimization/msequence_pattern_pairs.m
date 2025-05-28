clear all;

%% Parameters
file_max = 'optimized_max_ris_pattern_hex.txt';
file_min = 'optimized_min_ris_pattern_hex.txt';
num_patterns = 6;  % total number of patterns per file

% Read max patterns
fid_max = fopen(file_max, 'r');
max_patterns = cell(num_patterns, 1);
for i = 1:num_patterns
    line = strtrim(fgetl(fid_max));
    parts = regexp(line, '(!0x[0-9a-fA-F]+)', 'tokens');
    max_patterns{i} = parts{1}{1};
end
fclose(fid_max);

% Read min patterns
fid_min = fopen(file_min, 'r');
min_patterns = cell(num_patterns, 1);
for i = 1:num_patterns
    line = strtrim(fgetl(fid_min));
    parts = regexp(line, '(!0x[0-9a-fA-F]+)', 'tokens');
    min_patterns{i} = parts{1}{1};
end
fclose(fid_min);

%% Select pair
pos = 6;  % <-- Choose value from 1 to 6

max_pow_pattern = max_patterns{pos};
min_pow_pattern = min_patterns{pos};

% Display selected patterns
disp(['Selected Max pattern: ', max_pow_pattern]);
disp(['Selected Min pattern: ', min_pow_pattern]);

% Sequences for 3 RIS testing 
mseq3_1=[0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0];
mseq3_2=[1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1];
mseq3_3=[0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,0];

mseq=mseq3_1;
period=0.5e-3/200; % seconds
duration=50; % seconds
sleep_time=0;

ris=ris_init('COM22', 115200);   % initialize RIS
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