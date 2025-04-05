%{
This code generates an m-sequence for a RIS
%}

clear all

%% Variables
ris_idx=zeros(1,256);
ris_idx(121)=1;
ris_idx='!0x8000000000000000000000000000000000';

all_off='!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on='!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
high=all_off;
low=all_on;
period=0.5e-3/200; % seconds
duration=50; % seconds
sleep_time=0;


mseq=[0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0];

ris=ris_init('COM6', 115200);   % initialize RIS

writeline(ris, low);
% Get response
response = readline(ris);
fprintf("Response from setting a pattern: %s\n", response);

% ris_sequence(ris, high, low, mseq, period, duration, sleep_time)




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