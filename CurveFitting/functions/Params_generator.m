function Params_generator(arg)

%% Parameters Generator of Neuron models
% Model choices : 'Gerstner', 'Bruenl'


switch arg
    
    case 1
%% Gerstners
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
    dt = 1e-4;       % time step
    tmax = 1;        % simulation time
    T = dt:dt:tmax;  % time step vector
    tau_R = 2e-3;    % absolute refractory period
    Re_t = round(tau_R/dt);
    
    
    save Gerstner_params

    case 2
    %% iaf neuron
    Cm = 250e-12;   % membrane capacitance
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
    T = dt:dt:tmax;  % time step vector
    tau_R = 2e-3;    % absolute refractory period
    Re_t = round(tau_R/dt);
    
    
    save Mohammad_params
end