%% Set up the Transmitter and the Receiver
% Free up channel (added by Laura, comment lines if the code is running for the first time)
%release(tx)
%release(rx)

% Set up parameters and signals
%sampleRate = 200e3;
centerFreq = 2.42e9;
fRef = 80e3;
sampleRate = 10*fRef;
s1 = exp(1j*2*pi*20e3*[0:10000-1]'/sampleRate);  % 20 kHz
s2 = exp(1j*2*pi*40e3*[0:10000-1]'/sampleRate);  % 40 kHz
s3 = exp(1j*2*pi*fRef*[0:10000-1]'/sampleRate);  % 80 kHz
%s = s1 + s2 + s3;
%s=s3;
s = 0.6*s/max(abs(s)); % Scale signal to avoid clipping in the time domain

% Set up the transmitter
% Use the default value of 0 for FrequencyCorrection, which corresponds to
% the factory-calibrated condition
tx = sdrtx('Pluto', 'RadioID', 'usb:0', 'CenterFrequency', centerFreq, ...
           'BasebandSampleRate', sampleRate, 'Gain', 0, ...
           'ShowAdvancedProperties', true);
% Use the info method to show the actual values of various hardware-related
% properties
txRadioInfo = info(tx)
% Send signals
disp('Send 3 tones at 20, 40, and 80 kHz');
transmitRepeat(tx, s);

% Set up the receiver
% Use the default value of 0 for FrequencyCorrection, which corresponds to
% the factory-calibrated condition

numSamples = 1024*1024;
rx = sdrrx('Pluto', 'RadioID', 'usb:1', 'CenterFrequency', centerFreq, ...
           'BasebandSampleRate', sampleRate, 'SamplesPerFrame', numSamples, ...
           'OutputDataType', 'double', 'ShowAdvancedProperties', true);
rx.EnableQuadratureCorrection = true;

% Use the info method to show the actual values of various hardware-related
% properties
rxRadioInfo = info(rx)



%% Receive and Visualize Signal

disp(['Capture signal and observe the frequency offset' newline])
receivedSig = rx();
receivedSig_ac=receivedSig-mean(receivedSig); % removes mean
rx_corrected = comm.IQImbalanceCompensator();
rxData = rx_corrected(receivedSig_ac); % corrects I/Q imbalance
%receivedSig_norm=receivedSig./abs(receivedSig);

% Constellation plot before freq offset correction
figure(1)
rx_re=real(rxData);
rx_im=imag(rxData);
plot(rx_re(1:12:end-12),rx_im(6:12:end),'.', 'LineWidth',2)
xlim([-2 2])
ylim([-2 2])
grid on
grid minor

% Find the tone that corresponds to the 80 kHz transmitted tone
y = fftshift(abs(fft(receivedSig)));
[~, idx] = findpeaks(y,'MinPeakProminence',max(0.5*y));
fReceived = (max(idx)-numSamples/2-1)/numSamples*sampleRate;

% Plot the spectrum
sa = spectrumAnalyzer('SampleRate', sampleRate);
sa.Title = sprintf('Tone Expected at 80 kHz, Actually Received at %.3f kHz', ...
                   fReceived/1000);
receivedSig = reshape(receivedSig, [], 16); % Reshape into 16 columns
for i = 1:size(receivedSig, 2)
    sa(receivedSig(:,i));
end


%% Estimate and Apply the Value of FrequencyCorrection

rx.FrequencyCorrection = (fReceived - fRef) / (centerFreq + fRef) * 1e6;
msg = sprintf(['Based on the tone detected at %.3f kHz, ' ...
               'FrequencyCorrection of the receiver should be set to %.4f'], ...
               fReceived/1000, rx.FrequencyCorrection);
disp(msg);
rxRadioInfo = info(rx)


%% Receive and Visualize Signal

% Capture 10 frames, but only use the last frame to skip the transient
% effects due to changing FrequencyCorrection
disp(['Capture signal and verify frequency correction' newline])
for i = 1:10
    receivedSig = rx();
end

receivedSig_oneraw=receivedSig;
receivedSig_norm=receivedSig_oneraw./abs(receivedSig_oneraw);

% Find the tone that corresponds to the 80 kHz transmitted tone
% fReceived2 should be very close to 80 kHz
y = fftshift(abs(fft(receivedSig)));
[~,idx] = findpeaks(y,'MinPeakProminence',max(0.5*y));
fReceived2 = (max(idx)-numSamples/2-1)/numSamples*sampleRate;

% Plot the spectrum
sa.Title = '3 Tones Received at 20, 40, and 80 kHz';
receivedSig = reshape(receivedSig, [], 16); % Reshape into 16 columns
for i = 1:size(receivedSig, 2)
    sa(receivedSig(:,i));
end
msg = sprintf('Tone detected at %.3f kHz\n', fReceived2/1000);
disp(msg);

% Constellation plot after freq offset correction
figure(3)
rx_re=real(receivedSig_norm);
rx_im=imag(receivedSig_norm);
plot(rx_re(1:12:end-12),rx_im(6:12:end),'.', 'LineWidth',2)
%plot(rx_re(8:10:end),rx_im(1:10:end-10),'.', 'LineWidth',2)
xlim([-2 2])
ylim([-2 2])
grid on
grid minor


% Release the radios %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
release(tx);
release(rx);


%{
%% Change FrequencyCorrection of the Transmitter

disp(['Change the FrequencyCorrection property of the transmitter to 1 to ' ...
      'simulate the effect that the transmitter''s oscillator has drifted'])
tx.FrequencyCorrection = 1; % 1 ppm
txRadioInfo = info(tx)
tx.transmitRepeat(s);

%% Receive and Visualize Signal

% Capture 10 frames, but use the last frame only to skip the transient
% effects due to changing FrequencyCorrection
disp(['Capture signal and observe the frequency offset' newline])
for i = 1:10
    receivedSig = rx();
end

% Find the tone that corresponds to the 80 kHz transmitted tone
% fReceived3 will not be close to 80 kHz because tx.FrequencyCorrection
% has been changed
y = fftshift(abs(fft(receivedSig)));
[~,idx] = findpeaks(y,'MinPeakProminence',max(0.5*y));
fReceived3 = (max(idx)-numSamples/2-1)/numSamples*sampleRate;

% Plot the spectrum
sa.Title = sprintf('Tone Expected at 80 kHz, Actually Received at %.3f kHz', ...
                   fReceived3/1000);
receivedSig = reshape(receivedSig, [], 16); % Reshape into 16 columns
for i = 1:size(receivedSig, 2)
    sa(receivedSig(:,i));
end


%% Estimate and Apply the Value of FrequencyCorrection

rxRadioInfo = info(rx);
currentPPM = rxRadioInfo.FrequencyCorrection;
ppmToAdd = (fReceived3 - fRef) / (centerFreq + fRef) * 1e6;
rx.FrequencyCorrection = currentPPM + ppmToAdd + currentPPM*ppmToAdd/1e6;
msg = sprintf(['Based on the tone detected at %.3f kHz, ' ...
               'FrequencyCorrection of the receiver should be changed from %.4f to %.4f'], ...
               fReceived3/1000, currentPPM, rx.FrequencyCorrection);
disp(msg);
rxRadioInfo = info(rx)



%% Receive and Visualize Signal

% Capture 10 frames, but use the last frame only to skip the transient
% effects due to changing FrequencyCorrection
disp(['Capture signal and verify frequency correction' newline])
for i = 1:10
    receivedSig = rx();
end

% Find the tone that corresponds to the 80 kHz transmitted tone
% fReceived4 should be very close to 80 kHz
y = fftshift(abs(fft(receivedSig)));
[~,idx] = findpeaks(y,'MinPeakProminence',max(0.5*y));
fReceived4 = (max(idx)-numSamples/2-1)/numSamples*sampleRate;

% Plot the spectrum
sa.Title = '3 Tones Received at 20, 40, and 80 kHz';
receivedSig = reshape(receivedSig, [], 16); % Reshape into 16 columns
for i = 1:size(receivedSig, 2)
    sa(receivedSig(:,i));
end
msg = sprintf('Tone detected at %.3f kHz', fReceived4/1000);
disp(msg);

% Release the radios
release(tx);
release(rx);

%}