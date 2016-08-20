%{
This funciton fits the simplified aeif analystical solution of F-I curve 
(with absolute refractory period)

This model genereate firing rate of simplified aeif model.
The exponential term of the spike-evoked adaptation is omitted,
and the subthreshold adaptation is kept also discarded.
The only difference is it doesn't allow the use of noise in stimulus.


to data of Gerstner's Aeif model using :
 
Lsqcurvefit

to optimize parameters in LIF model

Result: 
 tau_R: 0.001714 , tau: 0.045254 , Vt :0.023825 


% other alternative optimization methods

lsqnonlin

lsqnonneg

funcitons

%}

clear
% Add search path to import the requred functions 
% genpath search the subfolders under the path,too.
PATH = genpath('/Users/pohsuanhuang/Desktop/Curriculum Section/2015WS/Lab rotation_Martin_Giese/Nest_code/mohammad/functions');
addpath(PATH);

fprintf('Generate FR...');

[FR,I_f]=aeif_firingrate_func() ; % generating Aeif Firing Rate
Params_generator(1);

%load parameters of Gerstner Model.
load Gerstner_params.mat


%% Analytical solution of LIF F-I curve

% Laod parameters of LIF, to overwrite the params of Gerstners'
Cm = 250e-12;   % membrane capacitance
EL = -70e-3;    % leak reversal
Vr = EL;        % reset potential

fun_ref =@(x,I_f)( x(1) + x(2)*log(   ((I_f-x(4))* x(2)/Cm)./(x(2)/Cm*(I_f-x(4))-(x(3)-Vr))  )).^-1; % Hz 

%% fit parameters with lsqcurvefit 
 

xdata = I_f;

ydata = FR ;

% Using Gerstner parameter values as initial values.
a_0 = 0; % initial value of adaptive current
x_0 = [tau_R,taum,Vt,a_0]; 

% fit_paras = lsqcurvefit(fun_ref,x_0,xdata,ydata) ;
fit_paras = lsqcurvefit(fun_ref,x_0,xdata,ydata) ;
fit_paras = real(fit_paras);
save fit_paras
info  = fprintf('Result: \n tau_R: %f , tau: %f , Vt :%f , a_0 : %f \n', fit_paras);

%% Plot the F-I curves

close all
figure(1)
% analytical solution
plot(I_f*1e9, fun_ref(fit_paras,I_f),'b');

hold on

% generatring FR of simplified Aeif
FR_LIF = LIF_FIcurve_function(fit_paras(1),fit_paras(2),fit_paras(3),I_f);

% LIF simulation with optimized paras
plot(I_f*1e9,FR_LIF,'m+')

%  Gerstner simulation
plot(I_f*1e9, FR,'r.')

%ylim([0, 1/tau_R])
ylim([0,300])


ylabel('Firing Rate Hz')
%xlabel(['Current amplitude is multipled by I_0 = ',num2str(I0),' would be ', num2str(M(end)*I0),'(A)'])
xlabel('Current amplitude [nA]')
% xlabel('Current amplitude [A]')
legend('LIF_opt','LIF sim','Gerstner sim','Location','SouthEast')
title('f-I curve(aEIF)')

%% plot the membrane potential of I_e = 1800 pA
I_e = 1.8 ;%nA

[V_1,FR1] = LIF_V_m_func(fit_paras(1),fit_paras(2),fit_paras(3),I_e,dt);
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




