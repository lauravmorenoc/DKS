clear all; clc;

%% ─────────────────────────── Parameters ───────────────────────────
file_max = 'optimized_max_ris_pattern_hex.txt';
file_min = 'optimized_min_ris_pattern_hex.txt';

% Baseline patterns (all-OFF / all-ON)
all_off = '!0x0000000000000000000000000000000000000000000000000000000000000000';
all_on  = '!0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF';

num_patterns = 9;          % lines per file = positions 1 … 6
pos          = 5;          % CHOOSE position you want (1…6)
new_scheme = false;      % <-- false ⇒ run baseline (all_off / all_on)


%% ────────────────────── Read patterns from files ──────────────────
[maxA,maxB] = read_two_patterns(file_max,num_patterns);
[minA,minB] = read_two_patterns(file_min,num_patterns);

% pick the pair for the desired position
max_pow_A = maxA{pos};   max_pow_B = maxB{pos};
min_pow_A = minA{pos};   min_pow_B = minB{pos};

fprintf("Max  pattern (A/B): %s | %s\n", max_pow_A, max_pow_B);
fprintf("Min  pattern (A/B): %s | %s\n", min_pow_A, min_pow_B);

%% ───────────────────────── M-sequence etc. ─────────────────────────
mseq3_1 = [0 0 0 0 1 0 1 0 1 0 0 0 0 1 1 0];
mseq = mseq3_1;

period    = 0.5e-3 / 200;   % symbol duration  (s)
%period    = 1;   % symbol duration  (s)
duration  = 200;             % total play time  (s)
sleep_time= 0;              % pause between loops


%% ───────────────────── Initialise both RIS boards ─────────────────
risA = ris_init('COM18',115200);
risB = ris_init('COM19',115200);
ris   = [risA,risB];        


%% ───────────────────── Choose what to transmit ────────────────────
if new_scheme
    disp('Using optimized patterns')
    high = {max_pow_A, max_pow_B};
    low  = {min_pow_A, min_pow_B};
else
    disp('Using baseline')
    high = {all_off,   all_off}; 
    low  = {all_on,    all_on};
end

%% ───────────────────────── Run the sequence ───────────────────────
ris_sequence_dual(ris, high, low, mseq, period, duration, sleep_time);
%writeline(risA, all_on);
%writeline(risB, all_on);



%% ─────────────────────────── Clean-up ─────────────────────────────
for h = ris,  try delete(h); catch, end, end



% =========================  FUNCTIONS  =============================


function [patA,patB] = read_two_patterns(filename,N)
% Read N lines and return two cell arrays with the A & B patterns.
    patA = cell(N,1);  patB = cell(N,1);
    fid  = fopen(filename,'r');
    for k = 1:N
        line   = strtrim(fgetl(fid));
        toks   = regexp(line,'(!0x[0-9a-fA-F]+)','tokens');
        patA{k}= toks{1}{1};   % first pattern on the line
        patB{k}= toks{2}{1};   % second pattern on the line
    end
    fclose(fid);
end

% -------------------------------------------------------------------
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

% -------------------------------------------------------------------
function ris_sequence_dual(ris, high, low, seq, period, duration, sleep_time)
% ris  : vector of serialport handles [risA , risB]
% high : cell array {'!0x..','!0x..'}   pattern for symbol 1
% low  : cell array {'!0x..','!0x..'}   pattern for symbol 0
%
    assert(numel(ris)==2,"Expecting exactly two RIS handles");
    assert(numel(high)==2 && numel(low)==2,"high/low must have two elements");

    t0 = tic;
    while toc(t0) < duration
        for s = seq
            if s==0
                patPair = low;
            elseif s==1
                patPair = high;
            else
                error('Sequence symbols must be 0 or 1');
            end
            % send to both boards
            for d = 1:2
                writeline(ris(d), patPair{d});
                readline(ris(d));        % read / discard echo
            end
            pause(period);
        end
        pause(sleep_time);
    end
end
