function mseq = generate_msequence(m)
    % Generates an m-sequence (maximum-length sequence) using an LFSR.
    %
    % Parameters:
    %   m (int): Order of the LFSR (register length)
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