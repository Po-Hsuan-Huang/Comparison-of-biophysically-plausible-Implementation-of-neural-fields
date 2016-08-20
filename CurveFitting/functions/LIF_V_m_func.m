%{
   Leaky Integrate and Fire model

   This script compute the F-I curve of LIF model.
   For the basic knowledge of integrate and
   fire neuron model, check Gerstner's website, chapter 1.1.

   
   The numerical method used is Euler method. Small step (dt) is required to 
   generate accurate firing rate.

   
   
   This script is a modification of Mohammad's code.

   Last modifed : P.S.Huang Dec 4  2015

%}

function  [v,FR] =LIF_V_m(Tau_R,Tau,V_t,I_e,dt)

Cm = 250e-12;   % membrane capacitance
tau = Tau;
tau_R = Tau_R;
R = tau/Cm;
% gL = 30e-9;     % leak conductance
EL = -70e-3;    % leak reversal
Vt =V_t;  % threshold potential
% Vp = 20e-3;     % peak potential
Vr = EL;        % reset potential
% Dt = 2e-3;      % rise slope factor
% a = 4e-9;       % 'sub-threshold' adaptation conductance
% b = 0.0805e-9;  % 'sra' current increment
% taum = Cm/gL;   % membrane time constant
% tauw = 0.144;   % adaptation time constant (~Ca-activated K current inactivation)

dt = dt;     % time step
tmax = 1;        % simulation time
T = dt:dt:tmax;   % time step vector

Re_t = round(tau_R/dt);

I = zeros(1,length(T));                      % applied current vector
% I0 = 1e-9;                                % applied current
Isigma = 0*0.005;                            % noise fraction of above
% istart = round(0.1*tmax/dt);                 % time to start current pulse
% iend = length(t)-istart;                     % time to end


istart = floor(0.2*length(T))+1;
iend = floor(0.8*length(T));
I_0 = 1e-9;


% simulate

% w = zeros(1,length(t));             % adaptaion variable (~Ca-K current) 
    FN = zeros(1,length(T));
    I(istart:iend) = I_e*I_0;                % initialise current pulse
    v = EL*ones(1,length(T));   % membrane potential vector
    counter = 0;
    
    for t = Re_t:length(T)
        
        if sum(FN(1,t-Re_t+1:t)) == 0
            
            v(t) = v(t-1) + (dt/tau)*(-v(t-1) + R*I(t-1) + EL);
            
            if  v(t) >= Vt
                FN(1,t) = 1;
                counter = counter + 1 ;
                v(t) = Vr;                              % reset/update v and w   
        
            end
         else
            v(t) = Vr;
         end
    end
    
ISI = diff(T(FN==1));
FR = ISI(end)^-1;
% plot dynamics
%%
% figure(1)
% subplot(6,1,1:3)
% plot(t,v*1e3,'LineWidth',2)
% title('AdEx LIF neuron')
% ylabel('v (mV)')
% axis([0 tmax -75 -50])
% subplot(6,1,4:5)
% plot(t,w*1e9,'LineWidth',2,'color','r')
% ylabel('w (nA)')
% axis([0 tmax min(w)*1e9-0.02 max(w)*1e9+0.02])
% subplot(6,1,6)
% plot(t,I*1e9,'LineWidth',2,'color','b')
% xlabel('t (s)'); ylabel('I (nA)')
% axis([0 tmax min(I)*1e9-0.2 max(I)*1e9+0.2])
%%

% 
% figure(1)
% subplot(2,1,1)
% plot(T,v*1e3,'LineWidth',2)
% title('LIF neuron')
% ylabel('v (mV)')
% axis([0 tmax -75 -50])
% % subplot(6,1,4:5)
% % plot(t,w*1e9,'LineWidth',2,'color','r')
% % ylabel('w (nA)')
% % axis([0 tmax min(w)*1e9-0.02 max(w)*1e9+0.02])
% subplot(2,1,2)
% plot(T,I*1e9,'LineWidth',2,'color','r')
% xlabel('t (s)'); ylabel('I (nA)')
% axis([0 tmax min(I)*1e9-0.2 max(I)*1e9+0.2])





      
end