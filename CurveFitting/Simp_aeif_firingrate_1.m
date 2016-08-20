%{
Simplified Adaptive Exponential LIF model ('AdEx'/'aEIF', Brette & Gerstner (2005))
Calculating the firing rate for a single neuron, applying the different
current amplitude
Edition in this version :
1. without subthreshold adaptation.
   compensate the loss with Vt-> -50.6 mV ,and a compensation current
   2.1659e-10 in w2

2. plotting superposed comparison
3. setting V_m = ones(,)*EL
4. calculate firing rate with ISI
5. during absolute refractory period:
                v2(t) = Vr;
                w2(t) = w2(t-1) + dt/tauw*( - w2(t-1));
************************************************************************
Author : Mohammad Hovaidi Ardestani, Po-Hsuan Huang
Date and Place : 30.11.2015, CIN, Tuebingen
%}

tic


% model parameters (in V, A, S, s)
%{
Cm = 281e-12;   % membrane capacitance
gL = 30e-9;     % leak conductance
EL = -70.6e-3;  % leak reversal
Vt = -50.4e-3;  % threshold potential
Vp = 20e-3;     % peak potential
Vr = EL;        % reset potential
Dt = 2e-3;      % rise slope factor
a = 4e-9;       % 'sub-threshold' adaptation conductance
b = 0.0805e-9;  % 'sra' current increment
taum = Cm/gL;   % membrane time constant
tauw = 0.144;   % adaptation time constant (~Ca-activated K current inactivation)
%}

Cm = 0.25e-9;   % membrane capacitance
gL = 25e-9;     % leak conductance
EL = -70.6e-3;  % leak reversal
Vt = -50.6e-3;  % threshold potential
Vp = 20e-3;     % peak potential
Vr = EL;        % reset potential
Dt = 2e-3;      % rise slope factor
a = 4e-9;       % 'sub-threshold' adaptation conductance
b = 0.0805e-9;  % 'sra' current increment
taum = 10.0 *1e-3;   % membrane time constant
tauw = 0.144;   % adaptation time constant (~Ca-activated K current inactivation)
%}

dt = 1e-4;       % time step
tmax = 1;        % simulation time
T = dt:dt:tmax;   % time step vector
tau_R = 2e-3;
Re_t = tau_R/dt;


Isigma = 0*0.005;                            % noise fraction of above
istart = floor(0.5*length(T));
iend = floor(0.9*length(T));
TimeWindow = iend-istart;
I2 = zeros(1,length(T));                      % applied current vector

v2 = zeros(1,length(T)); v2(1) = EL;  % membrane potential vector
w2 = zeros(1,length(T));             % adaptaion variable (~Ca-K current) 
g = zeros(1,length(T));             % total synaptic conductance variables vector
E = zeros(1,length(T));             % synaptic reversal potentials vector

% 
m =0.8;
dm =0.1;
M = 0:dm:m;                         % Loop counter to calculate  the firing rate
I0 = 1e-9;                         % applied current

FiringNeurons = zeros(1,length(T));
FR2 = zeros(1, length(M));         % firing rate of simp_aeif neuron


I_f = zeros(1,length(M));
spike_num=zeros(1,length(M));
h = waitbar(0,'Please wait...');

for i = 1:length(M)   % apply voltage ramp
        
    counter = 0;
    FiringNeurons(:) = 0;
    v2 = EL*ones(1,length(T));           % membrane potential vector
    w2 = zeros(1,length(T));             % adaptaion variable (~Ca-K current) 
    g = zeros(1,length(T));             % total synaptic conductance variables vector
    E = zeros(1,length(T));             % synaptic reversal potentials vector
    I2(istart:iend) = M(i)*I0;           % initialise current pulse
    
    %% Filtered noise
    var = 0.003; % variance of the white noise
    noise = sqrt(var)* M(i)*I0*randn(1,iend-istart+1);
    
    Fs = 30e3;                        %Sampling frequency
    Nyquist = Fs/2;
    [a2,a1] = butter(2,[100/Nyquist,(2e3-1)/Nyquist]);
    [y] = filtfilt(a2,a1,noise);   % Zero-phase filtering helps preserve features
    
    %%
    I2(istart:iend) = M(i)*I0*ones(1,iend-istart+1); %+y;           % initialise current pulse with white noise

    I_f(1,i) = M(i)*I0;  
    
        for t = Re_t:length(T)
            if ((sum(FiringNeurons(t-Re_t+1:t))) == 0) % refractory period : no membrane potential change with 2ms window of activation.
               
    
                % updated value v(i) is used in synaptic current to avoid instability
                 v2(t) = v2(t-1) + dt/Cm*(gL*(EL-v2(t-1)) + gL*Dt*exp((v2(t-1)-Vt)/Dt) - w2(t-1) + I2(t-1));
    
%                  w(t) = w(t-1) + dt/tauw*(a*(v(t-1)-EL) - w(t-1)); %update adaptation conductance
                 compensation =   0; 2.1659e-10;  % compensate for the absence of subthreshold adaptation.
                 w2(t) = w2(t-1) + dt/tauw*( - w2(t-1))+ compensation;
                if v2(t) >= Vp                                % if spike
                    FiringNeurons(1,t) = 1;
                    spike_num(i)=spike_num(i)+1;
                    v2(t) = Vr;                              % reset/update v and w   
                    w2(t) = w2(t-1) + b;
                end
            else
                v2(t) = Vr;
                w2(t) = w2(t-1) + dt/tauw*( - w2(t-1));
            end
        end


            if spike_num(i)>1
                 Spiketime = find(FiringNeurons==1);
                 Spikeend= Spiketime(end-1:end);
                 FR2(1,i) = diff(T(Spikeend))^-1;
            else 
                 FR2(1,i) = 0.0; 
            end
                 
end
close(h)

toc;

%% Plot Dynamics
figure(1)
plot(I_f*1e9,FR2,'LineWidth',1.7)% firing rate of simplified aeif neuron
hold on
%plot(I_f*1e9,FR1,'.-','LineWidth',1.7,'MarkerSize',8.0)% firing rate of aeif neuron
hold off
ylim([0,400])
% axis([ min(I_f) max(I_f) 0 1.1])
ylabel('Firing Rate /Hz')
%xlabel(['Current amplitude is multipled by I_0 = ',num2str(I0),' would be ', num2str(M(end)*I0),'(A)'])
xlabel('Current amplitude [nA]')
% xlabel('Current amplitude [A]')
legend('FR_simplified aeif','FR_aeif','Location','southeast')
title('f-I curve(aEIF)')

figure(2)
subplot(211)
plot(T,1000*v2,'m') % firing rate of simplified aeif neuron
hold on
plot(T,1000*v1,'b') % firing rate of aeif neuron
ylabel(' mV')
xlabel(' ms')
ylim([-80,-40])
xlim([0.4,1.0])
hold off
legend('simp_aeif','aeif')


subplot(212)
plot(T,1000*w2,'m') % adaptive current of simplified neuron
hold on 
plot(T,1000*w1,'b') % adaptive current of aeif neuron
hold off
ylabel(' nA')
xlabel(' ms')
str = sprintf('injecting current %2.f pA', I_f(1,end)*1e12 );
legend(str,'Location','southeast')
xlim([0.4,1.0])
figure(3)
subplot(211)
plot(T,1000*I2,'m') % adaptive current of simplified neuron
subplot(212)
plot(T,1000*I1,'b') % adaptive current of aeif neuron
hold off
ylabel(' nA')
xlabel(' ms')
