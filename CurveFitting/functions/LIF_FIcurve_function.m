%  LIF model
% 26.11.2015

function  FR=LIF_FIcurve_function(Tau_R,Taum,Vt,I_f)
tic


Cm = 250e-12;   % membrane capacitance
tau = Taum;
tau_R = Tau_R;
R = tau/Cm;
% gL = 30e-9;     % leak conductance
EL = -70e-3;    % leak reversal
Vt = Vt;  % threshold potential
% Vp = 20e-3;     % peak potential
Vr = EL;        % reset potential

dt = 1e-4;     % time step
tmax = 1;        % simulation time
T = dt:dt:tmax;   % time step vector

Re_t = round(tau_R/dt);

I = zeros(1,length(T));                      % applied current vector
% I0 = 1e-9;                                % applied current


istart = floor(0.2*length(T))+1;
iend = floor(0.8*length(T));
FR = zeros(1,length(I_f));


% simulate

% w = zeros(1,length(t));             % adaptaion variable (~Ca-K current) 
spikeNum  = zeros(1,length(I_f));
for j = 1:length(I_f)
    FN = zeros(1,length(T));
    I(istart:iend) = I_f(j);                % initialise current pulse
    v = EL*ones(1,length(T));   % membrane potential vector
    counter = 0;
    
    for t = Re_t:length(T)
        
        if sum(FN(1,t-Re_t+1:t)) == 0
            
            v(t) = v(t-1) + (dt/tau)*(-v(t-1) + R*I(t-1) + EL);
            
            if  v(t) >= Vt
                counter = counter + 1 ;
                spikeNum(j)= spikeNum(j)+ 1;
                FN(1,t) = 1;
                v(t) = Vr;                              % reset/update v and w   
    
            end
         else
            v(t) = Vr;
         end
    end
    
    if counter>1
        ISI = diff(T(FN==1));
        FR(1,j) = ISI(end)^-1;
    else 
         FR(1,j) = 0.0; 
    end

end

toc;






end