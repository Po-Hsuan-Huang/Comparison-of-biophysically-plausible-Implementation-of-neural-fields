%  LIF model
% 26.11.2015


tic
clear

Cm = 250e-12;   % membrane capacitance
tau = 10e-3;
tau_R = 2e-3;
R = tau/Cm;
% gL = 30e-9;     % leak conductance
EL = -70e-3;    % leak reversal
Vt = -55e-3;  % threshold potential
% Vp = 20e-3;     % peak potential
Vr = EL;        % reset potential
% Dt = 2e-3;      % rise slope factor
% a = 4e-9;       % 'sub-threshold' adaptation conductance
% b = 0.0805e-9;  % 'sra' current increment
% taum = Cm/gL;   % membrane time constant
% tauw = 0.144;   % adaptation time constant (~Ca-activated K current inactivation)

dt = 1e-4;     % time step
tmax = 1;        % simulation time
t = dt:dt:tmax;   % time step vector

Re_t = tau_R/dt;

I = zeros(1,length(t));                      % applied current vector
% I0 = 1e-9;                                % applied current
Isigma = 0*0.005;                            % noise fraction of above
% istart = round(0.1*tmax/dt);                 % time to start current pulse
% iend = length(t)-istart;                     % time to end

Imax = 10;  %  nA
Imin =  0.0;
Istep = 0.1;
M = Imin:Istep:Imax;                           
I_e = 1e-9;  
I_f = M*I_e;
I0 = .5e-9;                                % applied current
I2 = .4e-9;
I1 = 0;

istart = floor(0.2*length(t))+1;
iend = floor(0.8*length(t));
TimeWindow = iend - istart;

FR = zeros(1,length(M));


% simulate

% w = zeros(1,length(t));             % adaptaion variable (~Ca-K current) 
spikeNum  = zeros(1,length(M));
for j = 1:length(M)
    FN = zeros(1,length(t));
    I(istart:iend) = I_f(j);                % initialise current pulse
    v = EL*ones(1,length(t));   % membrane potential vector
    counter = 0;
    
    for i = Re_t:length(t)
        
        if sum(FN(1,i-Re_t+1:i)) == 0
            
            v(i) = v(i-1) + (dt/tau)*(-v(i-1) + R*I(i-1) + EL);
            
            if  v(i) >= Vt
                counter = counter + 1 ;
                spikes(counter) = i*dt;
                spikeNum(j)= spikeNum(j)+ 1;
                FN(1,i) = 1;
                v(i) = Vr;                              % reset/update v and w   
                    
                
                if counter ==1 
                    ISI(j) =spikes(counter);
                else
                    ISI(j) = spikes(counter)-spikes(counter-1);
                end
            end
         else
            v(i) = Vr;
         end
    end
 FR(1,j) = (sum(FN(1,istart:iend)))/(TimeWindow*dt);

end

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

close all

% figure(1)
% subplot(6,1,1:5)
% plot(t,v*1e3,'LineWidth',2)
% title('LIF neuron')
% ylabel('v (mV)')
% axis([0 tmax -75 -50])
% % subplot(6,1,4:5)
% % plot(t,w*1e9,'LineWidth',2,'color','r')
% % ylabel('w (nA)')
% % axis([0 tmax min(w)*1e9-0.02 max(w)*1e9+0.02])
% subplot(6,1,6)
% plot(t,I*1e9,'LineWidth',2,'color','r')
% xlabel('t (s)'); ylabel('I (nA)')
% axis([0 tmax min(I)*1e9-0.2 max(I)*1e9+0.2])
% 
% 



figure(2)
y1 = (tau*log(   (I_f * R)./(R*I_f-(Vt-Vr))  )).^-1; % Hz 
plot(I_f*1e9,y1 ,'p')


hold on

y1 = ( tau_R + tau*log(   (I_f * R)./(R*I_f-(Vt-Vr))  )).^-1; % Hz 
plot(I_f*1e9,y1 ,'v' )


%plot(I_f*1e9,FR,'LineWidth',1.7) % less accurate

plot(I_f*1e9, 1./ISI ,'b*'); %use the last non-zero
ylim([0, 1/tau_R])
ylabel('Firing Rate')
xlabel('Current amplitude [nA]')
title('f-I curve')
legend('no-ref','ref','sim','Location','southeast')

      
toc;