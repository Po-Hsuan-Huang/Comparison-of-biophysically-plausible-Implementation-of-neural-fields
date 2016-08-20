% Simplified Adaptive Exponential LIF model ('AdEx'/'aEIF', Brette & Gerstner (2005))
% Calculating the firing rate for a single neuron, applying the different
% current amplitude
% ************************************************************************
% Author : Mohammad Hovaidi Ardestani, Po-Hsuan Huang
% Date and Place : 30.11.2015, CIN, Tuebingen

tic
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

Cm = 0.250e-9;   % membrane capacitance
gL = 25e-9;     % leak conductance
EL = -70.6e-3;  % leak reversal
Vt = -50.4e-3;  % threshold potential
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
I1 = zeros(1,length(T));                      % applied current vector

v1 = zeros(1,length(T)); v1(1) = EL;  % membrane potential vector
w1 = zeros(1,length(T));             % adaptaion variable (~Ca-K current) 
g = zeros(1,length(T));             % total synaptic conductance variables vector
E = zeros(1,length(T));             % synaptic reversal potentials vector

% 
m = 10;
dm = 0.1;
M = 0:dm:m;                         % Loop counter to calculate  the firing rate
I0 = 1e-9;                         % applied current

FiringNeurons = zeros(1,length(T));
FR1 = zeros(1, length(M));
spike_num=zeros(1,length(M));


I_f = zeros(1,length(M));

h = waitbar(0,'Please wait...');

for i = 1:length(M)
        
    counter = 0;
    FiringNeurons(:) = 0;
    v1 = EL*ones(1,length(T));           % membrane potential vector
    w1 = zeros(1,length(T));             % adaptaion variable (~Ca-K current) 
    g = zeros(1,length(T));             % total synaptic conductance variables vector
    E = zeros(1,length(T));             % synaptic reversal potentials vector
    I1(istart:iend) = M(i)*I0;           % initialise current pulse
    I_f(1,i) = M(i)*I0;  
    %% Filtered noise
    var = 0.003; % variance of the white noise
    noise = sqrt(var)* M(i)*I0*randn(1,iend-istart+1);

    Fs = 30e3;                        %Sampling frequency
    Nyquist = Fs/2;
    [a2,a1] = butter(2,[100/Nyquist,(2e3-1)/Nyquist]);
    [y] = filtfilt(a2,a1,noise);   % Zero-phase filtering helps preserve features
    %%
    I1(istart:iend) = M(i)*I0*ones(1,iend-istart+1); % +y;
    % initialise current pulse with white noise

        for t = Re_t:length(T)
            if ((sum(FiringNeurons(t-Re_t+1:t))) == 0) % refractory period : no membrane potential change with 2ms window of activation.
               
    
                                    % updated value v(i) is used in synaptic current to avoid instability
                  v1(t) = v1(t-1) + dt/Cm*(gL*(EL-v1(t-1)) + gL*Dt*exp((v1(t-1)-Vt)/Dt) - w1(t-1) + I1(t-1));
    
                  w1(t) = w1(t-1) + dt/tauw*(a*(v1(t-1)-EL) - w1(t-1)); %update adaptation conductance
                if v1(t) >= Vp                                % if spike
                    FiringNeurons(1,t) = 1;
                    spike_num(i)=spike_num(i)+1;
                    v1(t) = Vr;                              % reset/update v and w   
                    w1(t) = w1(t-1) + b;
                end
            else
                v1(t) = Vr;
                w1(t) = w1(t-1) + dt/tauw*(a*(v1(t-1)-EL) - w1(t-1)); %update adaptation conductance

            end
        end


            if spike_num(i)>1
                 Spiketime = find(FiringNeurons==1);
                 Spikeend= Spiketime(end-1:end);
                 FR1(1,i) = diff(T(Spikeend))^-1;
            else 
                 FR1(1,i) = 0.0; 
            end
            waitbar(i / length(M));
end
close(h)
toc;


%% Plot Dynamics
figure(1)
plot(I_f*1e12,FR1,'LineWidth',1.7);
axis([ min(I_f)*1e12 max(I_f)*1e12 0 max(FR1)])
% axis([ min(I_f) max(I_f) 0 1.1])
ylabel('Firing Rate [Hz]')
%xlabel(['Current amplitude is multipled by I_0 = ',num2str(I0),' would be ', num2str(M(end)*I0),'(A)'])
xlabel('Current amplitude [pA]')
% xlabel('Current amplitude [A]')
title('f-I curve(aEIF)')
%%
% figure(2)
% subplot(211)
% plot(T,1000*v1)
% ylabel(' mV')
% xlabel(' ms')
% ylim([-80,-40])
% str = sprintf('injecting current %1.f', I_f(1,end) );
% subplot(212)
% plot(T,I1)
% ylabel(' nA')
% xlabel(' ms')
% legend(str)