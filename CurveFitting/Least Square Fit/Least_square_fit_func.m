%{
This funciton fits the LIF analystical solution of F-I curve (with absolute refractory period)
to data of Gerstner's Aeif model at either stimulus onset, or stable state.
THis can be switched in aeif_firingrate_func.m.

Opitmization method:

 
Lsqcurvefit

to optimize parameters in LIF model

Result: 
 tau_R: 0.001714 , tau: 0.045254 , Vt :0.023825 
12/12/2015 for the unknown aeif parameter shown on the slides


Result: 
 tau_R: 0.001545 , tau: 0.029195 , Vt :0.003356 
13/1/2016 

% other alternative optimization methods

lsqnonlin

lsqnonneg

funcitons

%}
function [fit_paras,fit_error]= Least_square_fit_func(fit_paras)
% Add search path to import the requred functions 
% genpath search the subfolders under the path,too.
PATH = genpath('/Users/pohsuanhuang/Desktop/Curriculum Section/2015WS/Lab rotation_Martin_Giese/Nest_code/mohammad/functions');
addpath(PATH);
load FR_onset.mat
load I_f.mat
if ~exist( 'FR_onset', 'var') % if the varialbe FR_onset doenst exist, then create the data
    fprintf('Generate FR...');
    [FR_onset,I_f]=aeif_firingrate_func(false) ; % generating Aeif Firing Rate
     save( 'FR_onset.mat', 'FR_onset')

else
    % data already exists
    
end
Params_generator(1);

%load parameters of Gerstner Model.
load Gerstner_params.mat


%% Analytical solution of LIF F-I curve

% Laod parameters of LIF, to overwrite the params of Gerstners'
Cm = 250e-12;   % membrane capacitance
EL = -70e-3;    % leak reversal
Vr = EL;        % reset potential

%fun_ref =@(x,I_f)( x(1) + x(2)*log(   (I_f* x(2)/Cm )./((x(2)*I_f/Cm)-(x(3)-Vr))  )).^-1; % Hz 
%fun_ref =@(x,I_f)( tau_R + x(1)*log(   (I_f* x(1)/Cm )./((x(1)*I_f/Cm)-(x(2)-Vr))  )).^-1; % Hz 
fun_ref =@(x,I_f)( tau_R + x(1)*log(   (I_f* x(1)/Cm )./((x(1)*I_f/Cm)-(Vt-Vr))  )).^-1; % Hz 

%% fit parameters with lsqcurvefit 
 

xdata = I_f;

ydata = FR_onset ;

% Using Gerstner parameter values as initial values.
x_0 = fit_paras;

% fit_paras = lsqcurvefit(fun_ref,x_0,xdata,ydata) ;
[fit_paras, fit_error] = lsqcurvefit(fun_ref,x_0,xdata,ydata) ;
fit_paras = real(fit_paras);
info  = fprintf('Result: \n tau_R: %f , tau: %f , Vt :%f,a_0 : %f \n', fit_paras);
info  = fprintf('Fit_error: \n %f \n', fit_error);

%% Plot the F-I curves

close all
figure(1)
% analytical solution
plot(I_f*1e9, fun_ref(fit_paras,I_f),'b');

hold on

% generatring FR of LIF
%FR_LIF = LIF_FIcurve_function(fit_paras(1),fit_paras(2),fit_paras(3),I_f);
%FR_LIF = LIF_FIcurve_function(fit_paras(1),fit_paras(2),fit_paras(3) ,I_f);

%FR_LIF_best = LIF_FIcurve_function(0.001714, 0.045254,0.023825,I_f);
% LIF simulation with optimized paras
%plot(I_f*1e9,FR_LIF,'m+')

%  Gerstner simulation
plot(I_f*1e9, FR_onset,'r.')

%ylim([0, 1/tau_R])
ylim([0,400])


ylabel('Firing Rate Hz')
%xlabel(['Current amplitude is multipled by I_0 = ',num2str(I0),' would be ', num2str(M(end)*I0),'(A)'])
xlabel('Current amplitude [nA]')
% xlabel('Current amplitude [A]')
legend('LIF_opt','LIF sim','Gerstner sim','Location','SouthEast')
title('f-I curve(aEIF)')

%% plot the membrane potential of I_e = 1800 pA
if false  % plot or not
    
    I_e = 2.0;%nA
    %fit_paras=[0.001714,0.045254,0.023825] at stable state
    %fit_paras=[ tau_R: 0.002108 , tau: 0.016426 , Vt :-0.034982 ] at onset
    [V_1,FR1] = LIF_V_m_func(tau_R,fit_paras(1),fit_paras(2),I_e,dt);
    [V_2,FR2] = Aeif_V_m_func(I_e,dt);



    figure(2)
    subplot(2,1,1)
    plot(T*1000,V_1*1e3,'LineWidth',2)
    title('LIF neuron')
    ylabel('v (mV)')
    s = sprintf('FR : %.2f', FR1);
    legend(s)

    subplot(2,1,2)
    plot(T*1000,V_2*1e3,'LineWidth',2)
    title('Aeif neuron')
    ylabel('v (mV)')
    xlabel('ms')
    s = sprintf('FR : %.2f', FR2);
    legend(s)
end



