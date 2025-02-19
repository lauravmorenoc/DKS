%{
This code generates an m-sequence for a RIS
%}



%% Variables
all_off='!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on='!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';
high=all_off;
low=all_on;
period=0.001; % seconds
duration=50; % seconds
sleep_time=1; % seconds

% Generate m-sequence
m = 6;                          % LFSR order (m), from 1 to 6
mseq = generate_msequence(m);

ris=ris_init('COM6', 115200);   % initialize RIS

ris_sequence(ris, high, low, mseq, period, duration, sleep_time)




%% Deinitialization
%clear ris;



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

function mseq = generate_msequence(m)
    % Generates an m-sequence (maximum-length sequence) using an LFSR.
    %
    % Parameters:
    %   m (int): Order of the LFSR (register length).
    % Returns:
    %   mseq (1D array): Generated m-sequence of length 2^m - 1.

    % Length of the sequence
    N = 2^m - 1;

    % Primitive polynomials for different m values
    % These determine the feedback taps for the LFSR.
    primitivePolynomials = {
        [1 1],       % m = 1
        [1 1 1],     % m = 2
        [1 0 1 1],   % m = 3
        [1 0 0 1 1], % m = 4
        [1 0 0 0 1 1], % m = 5
        [1 0 0 0 0 1 1]  % m = 6
    };

    if m > length(primitivePolynomials)
        error('Primitive polynomial for m=%d is not available.', m);
    end

    % Get the corresponding primitive polynomial
    poly = primitivePolynomials{m};

    % Initialize LFSR with all ones
    lfsr = ones(1, m);
    mseq = zeros(1, N);

    % Generate the m-sequence
    for i = 1:N
        mseq(i) = lfsr(end);  % Output bit (last bit in LFSR)

        % Compute feedback (XOR of selected taps)
        feedback = mod(sum(lfsr(poly(2:end) == 1)), 2);

        % Shift LFSR and insert new bit
        lfsr = [feedback, lfsr(1:end-1)];
    end
end

function ris_sequence(ris, high, low, sequence, period, duration, sleep_time)
    relative_time=0;
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
                fprintf("Response from setting a pattern: %s\n", response);
                fprintf("Current pattern: %s\n", currentPattern);
                fprintf("Current symbol: %s\n", sequence(i));
                pause(period);
                time=time+period;
                %relative_time=relative_time+period;
            end
            pause(sleep_time)
        %{
        else
            fprintf("Sleeping");
            relative_time=relative_time+period;
            if (relative_time>2*sleep_time)
                relative_time=0;
            end
        end
        %}
    end
end