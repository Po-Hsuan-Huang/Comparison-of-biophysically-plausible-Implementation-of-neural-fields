% Simplified Adaptive Exponential LIF model ('AdEx'/'aEIF', Brette & Gerstner (2005))
% Calculating the firing rate for a single neuron, applying the different
% current amplitude
% ************************************************************************
% Author : Mohammad Hovaidi Ardestani, Po-Hsuan Huang
% Date and Place : 30.11.2015, CIN, Tuebingen

I_e = 0.9;  % in nano Amphere
dt= 0.1*1e-3; % in ms
%function [v,FR] = simp_Aeif_V_m(I_e,dt)

    Cm = 0.45e-9;   % membrane capacitance
    gL = 25e-9;     % leak conductance
    EL = -70.6e-3;  % leak reversal
    Vt = -50.4e-3;  % threshold potential
    Vp = 20e-3;     % peak potential
    Vr = EL;        % reset potential
    Dt = 2e-3;      % rise slope factor
    a = 0;       % 'sub-threshold' adaptation conductance
    b = 0.0805e-9;  % 'sra' current increment
    taum = Cm/gL;   % membrane time constant
    tauw = 0.144;   % adaptation time constant (~Ca-activated K current inactivation)
    dt=dt;       % time step
    tmax = 1;        % simulation time
    T = dt:dt:tmax;   % time step vector
    tau_R = 2e-3;
    Re_t = tau_R/dt;

    Isigma = 0*0.005;                            % noise fraction of above
    istart = floor(0.2*length(T));
    iend = floor(0.8*length(T));
    TimeWindow = iend-istart;
    I = zeros(1,length(T));                      % applied current vector

    v = EL*ones(1,length(T));  % membrane potential vector
    w = zeros(1,length(T));             % adaptaion variable (~Ca-K current) 
    g = zeros(1,length(T));             % total synaptic conductance variables vector
    E = zeros(1,length(T));             % synaptic reversal potentials vector


% 
                         % Loop counter to calculate  the firing rate
    I0 = 1e-9;                         % applied current

    FiringNeurons = zeros(1,length(T));
    spike_num=0;    
 
    %%
    I(istart:iend) = I_e*I0; % +y;
    % initialise current pulse with white noise

        for t = Re_t:length(T)

            if ((sum(FiringNeurons(t-Re_t+1:t))) == 0) % refractory period : no membrane potential change with 2ms window of activation.
               
    
                                    % updated value v(i) is used in synaptic current to avoid instability
                  v(t) = v(t-1) + dt/Cm*(gL*(EL-v(t-1)) - w(t-1) + I(t-1));
                 
                  w(t) = w(t-1) + dt/tauw*(a*(v(t)-EL) - w(t-1)); %update adaptation conductance
                if v(t) >= Vt                                % if spike
                    FiringNeurons(1,t) = 1;
                    spike_num=spike_num+1;
                    v(t) = Vr;                              % reset/update v and w   
                    w(t) = w(t-1) +b;
                end
                
            else
                v(t) = Vr;
                w(t) = w(t-1) + dt/tauw*(a*(v(t)-EL) - w(t-1)); %update adaptation conductance

            end
        end


           
   
ISI = diff(T(FiringNeurons==1));
FR = ISI(end)^-1;



%% Plot Dynamics
% %% figure(1)
% plot(I_f*1e9,FR1,'LineWidth',1.7)
% axis([ min(I_f)*1e9 max(I_f)*1e9 0 max(FR1)])
% % axis([ min(I_f) max(I_f) 0 1.1])
% ylabel('Firing Rate')
% %xlabel(['Current amplitude is multipled by I_0 = ',num2str(I0),' would be ', num2str(M(end)*I0),'(A)'])
% xlabel('Current amplitude [nA]')
% % xlabel('Current amplitude [A]')
% title('f-I curve(aEIF)')
%%
figure(2)
subplot(211)
plot(T,1000*v)
ylabel(' mV')
xlabel(' ms')
ylim([-80,-40])
str = sprintf('injecting current %1.f', I_e );
subplot(212)
plot(T,w)
ylabel(' nA')
xlabel(' ms')
legend(str)
