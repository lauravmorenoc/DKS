%{
This code generates an m-sequence for a RIS
%}

clear all

%% Variables

all_off='!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on='!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
high=all_off;
low=all_on;
period=0.003; % seconds # 0.5e-3/200
duration=50; % seconds
sleep_time=0;

% mseq=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]; % RIS 2
% mseq=[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]; % RIS 1
mseq1=[0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0];
% mseq2=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1];

% Sequences for 3 RIS testing 
% mseq3_1=[0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0];
% mseq3_2=[1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1];
% mseq3_3=[0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,0];
mseq=mseq1;

ris1=ris_init('COM16', 115200);   % initialize RIS 1
ris2=ris_init('COM17', 115200);   % initialize RIS 2
ris3=ris_init('COM18', 115200);   % initialize RIS 3
ris4=ris_init('COM19', 115200);   % initialize RIS 4
%ris_all = {ris1, ris2, ris3, ris4};
ris_all = {ris1, ris2, ris3, ris4};

currentPattern=low;

ris_sequence(ris_all, high, low, mseq, period, duration, sleep_time)

%{
writeline(ris1, currentPattern);
response = readline(ris1); % Get response
fprintf("Response from setting a pattern to RIS 1: %s\n", response);
writeline(ris2, currentPattern);
response = readline(ris2); % Get response
fprintf("Response from setting a pattern to RIS 2: %s\n", response);
writeline(ris3, currentPattern);
response = readline(ris3); % Get response
fprintf("Response from setting a pattern to RIS 3: %s\n", response);
writeline(ris4, currentPattern);
response = readline(ris4); % Get response
fprintf("Response from setting a pattern to RIS 4: %s\n", response);
%}

%ris_sequence(ris, high, low, mseq, period, duration, sleep_time)




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


function ris_sequence(ris_array, high, low, sequence, period, duration, sleep_time)
    time = 0;
    while (time < duration)
        for i = 1:length(sequence)
            switch sequence(i)
                case 0
                    currentPattern = low;
                case 1
                    currentPattern = high;
                otherwise
                    disp('Could not write sequence value, it must be either 0 or 1');
                    continue;
            end
            
            % Send pattern to all RIS devices
            for r = 1:length(ris_array)
                writeline(ris_array{r}, currentPattern);
            end
            fprintf("Current pattern: %s\n", currentPattern);
            % Read response from all RIS devices
            for r = 1:length(ris_array)
                if ris_array{r}.NumBytesAvailable > 0
                    response = readline(ris_array{r});
                    %fprintf("RIS %d response: %s\n", r, response);
                end
            end

            pause(period);
            time = time + period;
        end
        pause(sleep_time);
    end
end

