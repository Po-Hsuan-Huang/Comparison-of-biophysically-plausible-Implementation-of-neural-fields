%{
  LIF model
  This model genereate firing rate of simplified aeif model.
  The exponential term of the spike-evoked adaptation is omitted,
  and the subthreshold adaptation is kept also discarded.
  The only difference is it doesn't allow the use of noise in stimulus.

  Return:

  FR    firing rate
  I_f   injecting current ramp
  12.1.2016
%}
function  FR=simp2_aeif_FIcurve_func(Tau_R,Taum,Vt,a_0,I_f)
tic

      Params_generator(1);
      load 'Gerstner_params.mat';
%     Cm = 0.45e-9;   % membrane capacitance
%     gL = 25e-9;     % leak conductance
%     EL = -70.6e-3;  % leak reversal
%     Vt = Vt ;%-50.4e-3;  % threshold potential
%     Vp = 20e-3;     % peak potential
%     Vr = EL;        % reset potential
%     Dt = 2e-3;      % rise slope factor
%     a = 0;       % 'sub-threshold' adaptation conductance
%     b = 0.0805e-9;  % 'sra' current increment
%     taum = Taum;   % membrane time constant
%     tauw = 0.144;   % adaptation time constant (~Ca-activated K current inactivation)
%     dt=1e-4;       % time step
%     tmax = 1;        % simulation time
%     T = dt:dt:tmax;   % time step vector
%     tau_R = Tau_R; %2e-3;
%     Re_t = round(tau_R/dt);

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

% simulate

% w = zeros(1,length(t));   
% adaptaion variable (~Ca-K current) 
for j = 1:length(I_f)
    FN = zeros(1,length(T));
    I(istart:iend) = I_f(j);                % initialise current pulse
    v = EL*ones(1,length(T));   % membrane potential vector
    counter = 0;

    
    for t = Re_t:length(T)
        if ((sum(FiringNeurons(t-Re_t+1:t))) == 0) % refractory period : no membrane potential change with 2ms window of activation.


            % updated value v(i) is used in synaptic current to avoid instability
             v(t) = v(t-1) + dt/Cm*(gL*(EL-v(t-1)) - w(t-1) + I(t-1));

             w(t) = w(t-1) + dt/tauw*( - w(t-1)); %update adaptation conductance

            if v(t) >= Vp                                % if spike
                FiringNeurons(1,t) = 1;
                counter = counter + 1;
                FN(1,t) = 1;
                v(t) = Vr;                              % reset/update v and w   
                w(t) = w(t-1) + b;
            end
        else
            v(t) = Vr;
            w(t) = w(t-1) + dt/tauw*(a*(v(t-1)-EL) - w(t-1)); %update adaptation conductance

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