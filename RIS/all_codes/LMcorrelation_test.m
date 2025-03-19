mseq3_1=[0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0];
mseq3_2=[1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1];
mseq3_3=[0,0,1,1,0,1,1,1,1,0,1,1,0,1,0,0];

%
mseq1=[0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0];
mseq2=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1];
%mseq3=

% Orthogonality check
mseq_aux = mseq3_1.* mseq3_3;
last_xor = mod(sum(mseq_aux), 2);

if last_xor == 0
    disp('Orthogonal')
else
    disp('Not orthogonal')
end

% 1 with 2 orthogonal
% 1 with 3 orthogonal
% 2 with 3 not ortogonal