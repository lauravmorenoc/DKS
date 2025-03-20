
% Set up parameters and signals
centerFreq = 2.42e9;
fRef = 80e3;
sampleRate = 4*fRef;
s1 = exp(1j*2*pi*20e3*[0:10000-1]'/sampleRate);  % 20 kHz
s2 = exp(1j*2*pi*40e3*[0:10000-1]'/sampleRate);  % 40 kHz
s3 = exp(1j*2*pi*fRef*[0:10000-1]'/sampleRate);  % 80 kHz
s=(s3);
s = 0.6*s/max(abs(s)); % Scale signal to avoid clipping in the time domain

% Graph
cos_part=real(s);
sin_part=imag(s);

s_re=real(s);
s_im=imag(s);
%stem((s(1:12:end)))
plot(s_re(1:12:end-12),s_im(6:12:end),'.')
xlim([-2 2])
ylim([-2 2])
grid on