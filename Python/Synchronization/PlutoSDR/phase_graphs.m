filename = 'phasedata.xlsx';
%{
t1 = xlsread(filename,'A3:A22');
t2 = xlsread(filename,'E3:E22');
t3 = xlsread(filename,'I3:I22');
t4 = xlsread(filename,'M3:M22');
t5 = xlsread(filename,'Q3:Q22');
t6 = xlsread(filename,'U3:U22');

f_co1=xlsread(filename,'B3:B22');
f_co2=xlsread(filename,'F3:F22');
f_co3=xlsread(filename,'J3:J22');
f_co4=xlsread(filename,'N3:N22');
f_co5=xlsread(filename,'R3:R22');
f_co6=xlsread(filename,'V3:V22');

f_re1=xlsread(filename,'C3:C22');
f_re2=xlsread(filename,'G3:G22');
f_re3=xlsread(filename,'K3:K22');
f_re4=xlsread(filename,'O3:O22');
f_re5=xlsread(filename,'S3:S22');
f_re6=xlsread(filename,'W3:W22');
%}

figure
hold on
plot(t1,f_co1./1000, 'LineWidth', 2)
plot(t2,f_co2./1000, 'LineWidth', 2)
plot(t3,f_co3./1000, 'LineWidth', 2)
plot(t4,f_co4./1000, 'LineWidth', 2)
plot(t5,f_co5./1000, 'LineWidth', 2)
plot(t6,f_co6./1000, 'LineWidth', 2)
legend('t=0.2','t=0.05','t=0.025','t=0.1','t=0.5','t=0.05', 'Location', 'best')
title('Only SDRs, no RIS')
ylabel('freq offset (kHz)')
xlabel('time (s)')
grid on
grid minor
hold off