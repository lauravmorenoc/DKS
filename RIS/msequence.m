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
poly = [1 0 0 0 0 1 1];  % Primitive polynom para m=6
init_state1 = [1 0 0 0 0 1];  % Estado inicial para la secuencia m original
init_state2 = [1 1 0 0 0 1];  % Estado inicial diferente para la secuencia ortogonal
mseq = generate_msequence(m, poly, init_state1);
gold_seq = generate_gold_sequence(m, poly, init_state1, init_state2);

% Verify orthogonality through dot product
dot_product = sum((2 * mseq - 1) .* (2 * gold_seq - 1));
disp(['Doct product (must be around 0 for orthogonality: ', num2str(dot_product)]);

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

function mseq = generate_msequence(m, poly, init_state)
    % Generates an m-sequence (maximum-length sequence) using an LFSR.
    %
    % Parameters:
    %   m (int): Order of the LFSR (register length).
    %   poly (array): The primitive polynomial for feedback taps.
    %   init_state (array): Initial state of the LFSR (must be of length m).
    %
    % Returns:
    %   mseq (array): Generated m-sequence.

    N = 2^m - 1;  % Length of the m-sequence
    lfsr = init_state;  % Initialize LFSR with the initial state
    mseq = zeros(1, N);  % Output m-sequence
    
    for i = 1:N
        mseq(i) = lfsr(end);  % Output bit (last bit in LFSR)

        % Compute feedback using XOR of selected taps
        feedback = mod(sum(lfsr(poly(2:end) == 1)), 2);
        
        % Shift LFSR and insert new bit
        lfsr = [feedback, lfsr(1:end-1)];
    end
end

function gold_seq = generate_gold_sequence(m, poly, init_state1, init_state2)
    % Generates a Gold sequence that is orthogonal to the m-sequence generated

    % Generate the original m-sequence
    mseq1 = generate_msequence(m, poly, init_state1);
    
    % Generate a second m-sequence with a different initial state
    mseq2 = generate_msequence(m, poly, init_state2);
    
    % Generate the Gold sequence by XORing the two m-sequences
    gold_seq = mod(mseq1 + mseq2, 2);
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