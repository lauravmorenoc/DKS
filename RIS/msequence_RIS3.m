%{
This code generates an m-sequence for a RIS and an orthogonal m-sequence
for a second RIS
%}

clear all

%% Variables
all_off='!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on='!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
high=all_off;
low=all_on;
period=0.5e-3/200; % seconds
duration=10; % seconds
%sleep_time=1.024; % seconds
sleep_time=0;
time_in_between=period/4;

% m sequences
%mseq1=[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]; % RIS 1
%mseq2=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]; % RIS 2
%mseq1=[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1];
%mseq1=[0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0];
%mseq2=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1];

%seq_0=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]; 
%seq_1=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];

mseq1=ones(1,16);%[0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0];
mseq2=ones(1,16);%[1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1];
mseq3=[0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,0];

ris1=ris_init('COM6', 115200);   % initialize RIS 1
ris2=ris_init('COM7', 115200);   % initialize RIS 2
ris3=ris_init('COM8', 115200);   % initialize RIS 2

ris_two_seqs(ris1, ris2, ris3, high, low, mseq1, mseq2, mseq3, period, duration, sleep_time, time_in_between);   





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

function ris_two_seqs(ris1, ris2,  ris3, high, low, sequence1, sequence2,  sequence3, period, duration, sleep_time, time_in_between)
    time=0;
    while (time<duration)
        for i=1:length(sequence1)
            switch sequence1(i)
                case 0
                    currentPattern1=low;
                case 1
                    currentPattern1=high;
                otherwise
                    disp('Could not write sequence value (RIS2), it must be either 0 or 1')
            end
            switch sequence2(i)
                case 0
                    currentPattern2=low;
                case 1
                    currentPattern2=high;
                otherwise
                    disp('Could not write sequence value (RIS2), it must be either 0 or 1')
            end
             switch sequence3(i)
                case 0
                    currentPattern3=low;
                case 1
                    currentPattern3=high;
                otherwise
                    disp('Could not write sequence value (RIS3), it must be either 0 or 1')
            end
            writeline(ris1, currentPattern1);
            pause(time_in_between);
            writeline(ris2, currentPattern2);
            writeline(ris3, currentPattern3);
            % Get response
            response = readline(ris1);
            %fprintf("Response from setting a pattern to RIS 1: %s\n", response);
            response = readline(ris2);
            %fprintf("Response from setting a pattern to RIS 2: %s\n", response);
            response = readline(ris3);

            time=time+period;
            pause(period);
        end
        pause(sleep_time)
    end 
end