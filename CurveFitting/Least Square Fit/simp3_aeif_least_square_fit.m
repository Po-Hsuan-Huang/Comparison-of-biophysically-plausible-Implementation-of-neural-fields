%{


This funciton fits the adaptive integrate and fire neuron moedel 
suggested in Loreen Hertaeg et. al.'s paper

'An Approximation to the Aeif model allows fast and predictive fitting to physiological data.'

This model genereate onset F-I curve where the frequency is derived mathematically at the 
onset of the stimulating step current; and the stable F-I curve is derived
with the condition t = infinity.
  
different from the paper, the subthreshold adaptation is disabled by setting a to zero.

The script fits the following parameters:

tauw, Vr, gL, Vt, b with a=0 

taum is known from physiological measure, and therefore C is known, too.



 

Lsqcurvefit

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


%% Analytical solution of simp2_aeif F-I curve

% Laod parameters of LIF, to overwrite the params of Gerstners'
Cm = 250e-12;   % membrane capacitance
EL = -70e-3;    % leak reversal
Vr = EL;        % reset potential
% tau_R: 0.001545 , tau: 0.029195 , Vt :0.003356 

fun_ref =@(x,I_f)( x(1) + x(2)*log(   ((I_f+x(4))* x(2)/Cm)./(x(2)/Cm*(I_f+x(4))-(x(3)-Vr))  )).^-1; % Hz 

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
FR_LIF = simp2_aeif_FIcurve_func(fit_paras(1),fit_paras(2),fit_paras(3),fit_paras(4),I_f);

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
legend('simp2_aeif_opt','simp2_aeif sim','Gerstner sim','Location','NorthWest')
title('f-I curve(aEIF)')

%% plot the membrane potential of I_e = 1800 pA
I_e = 2.0 ;%nA

[V_1,FR1] = simp2_Aeif_V_m_func(fit_paras(1),fit_paras(2),fit_paras(3),fit_paras(4),I_e,dt);
[V_2,FR2] = Aeif_V_m_func(I_e,dt);



figure(2)
subplot(2,1,1)
plot(T*1000,V_1*1e3,'LineWidth',2)
title('simp2 aeif neuron')
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




