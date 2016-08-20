% Adaptive Exponential LIF model ('AdEx'/'aEIF', Brette & Gerstner (2005))
% Calculating the firing rate for a single neuron, applying the different
% current amplitude
% ************************************************************************
% Author : Mohammad Hovaidi Ardestani
% Date and Place : 16.02.2015, CIN, Tuebingen
clear
tic


% model parameters (in V, A, S, s)

% Cm = 281e-12;   % membrane capacitance
% gL = 30e-9;     % leak conductance
% EL = -70.6e-3;  % leak reversal
% Vt = -50.4e-3;  % threshold potential
% Vp = 20e-3;     % peak potential
% Vr = EL;        % reset potential
% Dt = 2e-3;      % rise slope factor
% a = 4e-9;       % 'sub-threshold' adaptation conductance
% b = 0.0805e-9;  % 'sra' current increment
% taum = Cm/gL;   % membrane time constant
% tauw = 0.144;   % adaptation time constant (~Ca-activated K current inactivation)

Cm = 0.45e-9;   % membrane capacitance
gL = 25e-9;     % leak conductance
EL = -70.6e-3;  % leak reversal
Vt = -50.4e-3;  % threshold potential
Vp = 20e-3;     % peak potential
Vr = EL;        % reset potential
Dt = 2e-3;      % rise slope factor
a = 4e-9;       % 'sub-threshold' adaptation conductance
b = 0.0805e-9;  % 'sra' current increment
taum = Cm/gL;   % membrane time constant
tauw = 0.144;   % adaptation time constant (~Ca-activated K current inactivation)
dt = 1e-4;       % time step
tmax = 1;        % simulation time
T = dt:dt:tmax;   % time step vector
tau_R = 2e-3;
Re_t = tau_R/dt;

Isigma = 0*0.005;                            % noise fraction of above
istart = floor(0.5*length(T));
iend = floor(0.8*length(T));
TimeWindow = iend-istart;
I = zeros(1,length(T));                      % applied current vector

v = zeros(1,length(T)); v(1) = EL;  % membrane potential vector
w = zeros(1,length(T));             % adaptaion variable (~Ca-K current) 
g = zeros(1,length(T));             % total synaptic conductance variables vector
E = zeros(1,length(T));             % synaptic reversal potentials vector


m = 1000;
dm = 1;
M = 0:dm:m;                         % Loop counter to calculate  the firing rate
I0 = .1e-9;                         % applied current

FiringNeurons = zeros(1,length(T));
FR = zeros(1, length(M));


I_f = zeros(1,length(M));

h = waitbar(0,'Please wait...');

for i = 1:length(M)
        
    counter = 0;
    FiringNeurons(:) = 0;
    v = zeros(1,length(T)); v(1) = EL;  % membrane potential vector
    w = zeros(1,length(T));             % adaptaion variable (~Ca-K current) 
    g = zeros(1,length(T));             % total synaptic conductance variables vector
    E = zeros(1,length(T));             % synaptic reversal potentials vector
    I(istart:iend) = M(i)*I0;           % initialise current pulse
    I_f(1,i) = M(i)*I0;  
    
        for t = Re_t:length(T)
            if ((sum(FiringNeurons(t-Re_t+1:t))) == 0) % refractory period : no membrane potential change with 2ms window of activation.
               
    
                                    % updated value v(i) is used in synaptic current to avoid instability
                 v(t) = v(t-1) + dt/Cm*(gL*(EL-v(t-1)) + gL*Dt*exp((v(t-1)-Vt)/Dt) - w(t-1) + I(t-1));
    
                 w(t) = w(t-1) + dt/tauw*(a*(v(t-1)-EL) - w(t-1)); %update adaptation conductance
    
                if v(t) >= Vp                                % if spike
                    FiringNeurons(1,t) = 1;
                    counter = counter + 1;
                    v(t) = Vr;                              % reset/update v and w   
                    w(t) = w(t-1) + b;
                end
            else
                v(t) = Vr;
            end
        end

%     FR(1,i) = sum(FiringNeurons(istart:iend))/(iend-istart);
%       FR(1,i) = sum(FiringNeurons(1,istart+1:iend))/TimeWindow;
        FR(1,i) = (sum(FiringNeurons(1,istart:iend)))/(TimeWindow*dt);
%     FR(1,i) = counter/(iend-istart);
    waitbar(i / length(M))
end
close(h)

toc;

%%% Plot Dynamics
%% figure(1)
plot(I_f*1e9,FR,'LineWidth',1.7)
axis([ min(I_f)*1e9 max(I_f)*1e9 0 max(FR)])
% axis([ min(I_f) max(I_f) 0 1.1])
ylabel('Firing Rate')
%xlabel(['Current amplitude is multipled by I_0 = ',num2str(I0),' would be ', num2str(M(end)*I0),'(A)'])
xlabel('Current amplitude [nA]')
% xlabel('Current amplitude [A]')
title('f-I curve(aEIF)')
%%


