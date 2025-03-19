mseq3_1=[0,0,0,0,1,0,1,0,1,0,0,0,0,1,1,0];
mseq3_2=[1,1,0,1,1,1,0,0,0,1,0,1,1,1,0,1];
mseq3_3=[0,0,1,1,0,1,1,1,1,0,1,1,0,1,0,0];
r1 = xcorr(mseq3_1); % 5
r2 = xcorr(mseq3_2); % 10
r3 = xcorr(mseq3_3); % 7
%r1 = xcorr(mseq1,mseq2);
subplot(3,1,1)
%stem(r1./max(r1))
stem(r1)
title('xcorr(mseq1)')
subplot(3,1,2)
%stem(r2./max(r1))
stem(r2)
title('xcorr(mseq2)')
subplot(3,1,3)
%stem(r3./max(r1))
stem(r3)
title('xcorr(mseq3)')