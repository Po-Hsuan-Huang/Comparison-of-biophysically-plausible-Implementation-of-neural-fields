 
%{
   

   Adaptive exponential integrate and fire model


   This script compute the F-I curve of aeif model
   (Brette and Gerstner 2005)without implementing refractory period.
   
   The numerical method used is Euler method. Small step (dt) is required to 
   generate accurate firing rate.

   Analytical solution of refrectory perioded and non-refr-perioded neuron model
   are plotted as blue and red respectively.
   
   This script is a modification of Mohammad's code.

   Last modifed : P.S.Huang Nov30 2015
    
   

%}






clear
tic


Cm = 250e-12;   % membrane capacitance
tau = 10e-3;
tau_R = 2e-3;
R = tau/Cm;
% gL = 30e-9;     % leak conductance
EL = -70e-3;    % leak reversal
Vt = -55e-3;  % threshold potential
Vp = 20e-3;     % spike speak voltage
% Vp = 20e-3;     % peak potential
Vr = EL;        % reset potential
% Dt = 2e-3;      % rise slope factor
% a = 4e-9;       % 'sub-threshold' adaptation conductance
% b = 0.0805e-9;  % 'sra' current increment
% taum = Cm/gL;   % membrane time constant
% tauw = 0.144;   % adaptation time constant (~Ca-activated K current inactivation)

dt = 1e-3;     % time step
tmax = 1;        % simulation time
t = dt:dt:tmax;   % time step vector

Re_t = tau_R/dt;

I = zeros(1,length(t));                      % applied current vector
% I0 = 1e-9;                                % applied current
Isigma = 0*0.005;                            % noise fraction of above
% istart = round(0.1*tmax/dt);                 % time to start current pulse
% iend = length(t)-istart;                     % time to end

m = 100;
dm = 1;
M = dm:dm:m;                         % Loop counter to calculate  the firing rate
I_e = .1e-9;

I0 = .5e-9;                                % applied current
I2 = .4e-9;
I1 = 0;
FirstStepI = floor(0.001/dt);          % Current Timing
SecondStepI = floor(0.2/dt);
ThirdStepI = floor(0.5/dt);            

istart = floor(0.2*length(t));
iend = floor(0.8*length(t));
TimeWindow = iend - istart;

% I(FirstStepI:SecondStepI) = I0;        % initialise current pulse
% I(SecondStepI:ThirdStepI) = I1;        % initialise current pulse
% I(ThirdStepI:end) = I2;

FiringNeurons = zeros(1,length(t));

counter = 0;

% set alternative neuron behaviour according to flag


% simulate

v = zeros(1,length(t)); v(1) = EL;  % membrane potential vector
w = zeros(1,length(t));             % adaptaion variable (~Ca-K current) 
% g = zeros(1,length(t));             % total synaptic conductance variables vector
% E = zeros(1,length(t));             % synaptic reversal potentials vector

% I(istart:iend) = I0;                % initialise current pulse

% sptimes = zeros(1,length(t)-1);     % store spiketimes
I_f = zeros(1,length(M));
spike_num=zeros(1,length(M));
for j = 1:length(M)
    
    FiringNeurons(1,:) = 0;
    I(istart:iend) = M(j)*I_e;                % initialise current pulse
    I_f(1,j) = M(j)*I_e;  
    v = ones(1,length(t))*EL;
    v(1) = EL;  % membrane potential vector
    
    for i = Re_t:length(t) 
       
        if (sum(FiringNeurons(1,i-Re_t+1:i)) == 0)
    
                %     In = I0*Isigma*randn();         % noisy input term
                %     g(i-1) = g(i-1);                % total conductance variable (add syaptic conductances g_i here!)
                %     E(i-1) = g(i-1)*E(i-1);             % g-weighted average reversal potential (add weighted g_i*E_i here)

                                                    % updated value v(i) is used in synaptic current to avoid instability
                        v(i) = v(i-1) + dt/tau*(-v(i-1) + R*I(i-1) + EL);

                %     w(i) = w(i-1) + dt/tauw*(a*(v(i-1)-EL) - w(i-1)); %update adaptation conductance


                        if (v(i) >= Vt)                              % if spike
                %       FiringNeurons(1,t) = 1; 
                        spike_num(j) =spike_num(j)+1 ;
                        counter = counter + 1 ;
                        FiringNeurons(1,i) = 1;
                %         ip = i-1 + (Vp-v(i-1))/(v(i)-v(i-1));   % estimate spike time
                %         sptimes(i-1) = ip*dt;
                        v(i) = Vp;                              % reset/update v and w   
                %         w(i) = w(i-1) + (ip-i+1)*dt/tauw*(a*(v(i-1)-EL) - w(i-1)) + b;
                %       w(i) = w(i) + (i-ip)*dt/tauw*(a*(v(i)-EL) - w(i)); %update adaptation conductance
                        end

                %     I(i-1) = I(i-1)+In;     % store applied I for plotting

        else
             
                 v(i) = Vr;
                 
        end
            % FR(1,i) = counter/(iend-istart);
            % FR(1,j) = (sum(FiringNeurons(1,istart:iend)))/(TimeWindow*dt);
            if spike_num(j)>1
                 Spiketime = find(FiringNeurons==1);
                 Spikeend= Spiketime(end-1:end);
                 FR(1,j) = diff(t(Spikeend))^-1;
            else 
                 FR(1,j) = 0.0; 
            end
                 
    end
end

% spikes = sptimes(sptimes>0);       % condense spike times vector


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
%% analytical F-I curve without absolute refractory period


     

figure(2)
close all
y1 = (tau*log(   (I_f * R)./(R*I_f-(Vt-Vr))  )).^-1; % Hz 
plot(I_f*1e9,y1 )

hold on

y1 = ( tau_R + tau*log(   (I_f * R)./(R*I_f-(Vt-Vr))  )).^-1; % Hz 
plot(I_f*1e9,y1 )
%%
%figure(2)
%plot(spike_num,'LineWidth',1.7)
 plot(I_f*1e9 ,FR,'LineWidth',1.7)
axis([ min(I_f)*1e9 max(I_f)*1e9 0 max(FR)])
 % axis([ min(I_f) max(I_f) 0 1.1])
 ylabel('Firing Rate')
% %xlabel(['Current amplitude is multipled by I_0 = ',num2str(I0),' would be ', num2str(M(end)*I0),'(A)'])
 xlabel('Current amplitude [nA]')
% % xlabel('Current amplitude [A]')
 title('f-I curve')
 
 legend('no-ref','ref','sim')

toc;