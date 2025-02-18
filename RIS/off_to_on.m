%% Variables
all_off='!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on='!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
control_bool=true;


%% Initialization

% Get a new RIS object from serial port
ris = serialport('COM6', 115200);

% Clear input buffer
pause(0.1);
while ris.NumBytesAvailable > 0
    readline(ris);
    pause(0.1);
end

%% Control

for i=1:1000
    if(control_bool)
        currentPattern=all_off;
    else
        currentPattern=all_on;
    end
    writeline(ris, currentPattern);
    % Get response
    response = readline(ris);
    fprintf("Response from setting a pattern: %s\n", response);
    fprintf("Current pattern: %s\n", currentPattern);
    pause(3);
    control_bool=~control_bool;
    
end

%% Deinitialization
clear ris;