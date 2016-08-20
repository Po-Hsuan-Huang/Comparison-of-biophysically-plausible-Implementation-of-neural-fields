%{
   This model genereate firing rate of simplified aeif model.
   The exponential term of the near spike characteristic is omitted,
   but the adaptation machenism remains untouched as in the original model.
   The only difference is it doesn't allow the use of noise in stimulus.

   Return:

   FR    firing rate
   I_f   injecting current ramp
%}

function  [FR,I_f]=simp_aeif_firingrate_func()
%% Simulate Gerstner's Aeif Model
    tic

    Params_generator(2);
    load Gerstner_params.mat
    a =0;
    v = zeros(1,length(T)); v(1) = EL;  % membrane potential vector
    w = zeros(1,length(T));             % adaptaion variable (~Ca-K current) 
    g = zeros(1,length(T));             % total synaptic conductance variables vector
    E = zeros(1,length(T));             % synaptic reversal potentials vector



    istart = floor(0.2*length(T));
    iend = floor(0.8*length(T));
    TimeWindow = iend-istart;
    I = zeros(1,length(T));                      % applied current vector

   
    m = 10;
    dm = 0.1;
    M = 0.0:dm:m;                         % Loop counter to calculate  the firing rate
    I0 = 1e-9;                         % applied current (Amphere)

    FiringNeurons = zeros(1,length(T));
    FR = zeros(1, length(M));


    I_f = zeros(1,length(M));

    wait_bar = waitbar(0,'Please wait...');


    for i = 1:length(M)

        counter = 0;
        FiringNeurons(:) = 0;
        v = ones(1,length(T))* EL;  % membrane potential vector
        w = zeros(1,length(T));             % adaptaion variable (~Ca-K current) 
        g = zeros(1,length(T));             % total synaptic conductance variables vector
        E = zeros(1,length(T));             % synaptic reversal potentials vector
        I(istart:iend) = M(i)*I0;           % initialise current pulse
        I_f(1,i) = M(i)*I0;  

            for t = Re_t:length(T)
                if ((sum(FiringNeurons(t-Re_t+1:t))) == 0) % refractory period : no membrane potential change with 2ms window of activation.


                                        % updated value v(i) is used in synaptic current to avoid instability
                     v(t) = v(t-1) + dt/Cm*(gL*(EL-v(t-1)) - w(t-1) + I(t-1));

                     w(t) = w(t-1) + dt/tauw*(a*(v(t-1)-EL) - w(t-1)); %update adaptation conductance

                    if v(t) >= Vp                                % if spike
                        FiringNeurons(1,t) = 1;
                        counter = counter + 1;
                        v(t) = Vr;                              % reset/update v and w   
                        w(t) = w(t-1) + b;
                    end
                else
                    v(t) = Vr;
                    w(t) = w(t-1) + dt/tauw*(a*(v(t-1)-EL) - w(t-1)); %update adaptation conductance

                end
            end

    if counter>1
         Spiketime = find(FiringNeurons==1);
         Spikeend= Spiketime(end-2:end-1);
         FR(1,i) = mean(diff(T(Spikeend)))^-1;
    else 
         FR(1,i) = 0.0; 
    end
        waitbar(i / length(M))
    end

        close(wait_bar);
    toc;

