% recursive optimization method.
clear
% given the initial fit_paras as AdEx models values
Params_generator(1);
load Gerstner_params.mat

a0 = 0.0;
fit_paras = [tau_R, taum, Vt, a0];

% given initial error value 

fit_error1 = Inf; 
fit_error2 = Inf;

%Conditioning the optimization loop, the for loop breaks when
% error1 and error2 are both smaller than the tolerance

tolerance = 1e5;
for i = 1:10
    if (fit_error1 <tolerance) && (fit_error2 <  tolerance)
        save fit_paras
        print('\n\n\n\n\n\n\n\n\n\n')
        print('optimization finished.')
        info  = sprintf('Result: \n tau_R: %f , tau: %f , Vt :%f, a0: %f \n', fit_paras);
        break;
    else
        % feed the initial paras into stable firing rate F-I fit funcion

        [fit_paras, fit_error1]=simp2_aeif_least_square_fit_func(fit_paras);

        % feed the  fit_paras into onset firing rate F-I fit function

        [fit_paras, fit_error2]= Least_square_fit_func(fit_paras);
    end
end
%%

    %print('optimization failed') 
    info = sprintf( ' \n fit_error1 = %f, fit_error2 = %f', fit_error1,fit_error2)
    