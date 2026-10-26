function [beta, d_beta, closest_models] = getDbetas(st, windows, deltaT, order, check_fit, quant)
     %windows is a matrix of the form [[start1, stop1] ; [start2, stop2] ;
     %... ; [startN, stopN]] specifying the time windows to be considered
     %st is the list of spike times
     %deltaT is the time bin length
     %order is the order of the AR model
     %if check_fit is true, then getDbetas spits out a plot comparging the 
     %avalanche distributions, power spectra, DFA fluctuation functions,
     %and activity distributions of the AR fit and the experimental data 
     %quant is the avalanche threshold
     [varphi, varNoise] = myYuleWalker(st, windows, deltaT, order);
     beta = 2:2:(2*order);
     d_beta = zeros(size(beta));
     closest_models = zeros(length(beta), order);
     for i = 1:length(beta)
         disp(i);
         [d_beta(i), curr] = getFixedPointDistance(order, beta(i), varphi);
         closest_models(i, :) = curr';
     end

     if check_fit
         pw_time = 200; %window length for pwelch
         [sizes, durations, sizesAR, durationsAR, lengths, F, FAR, freq, pxx_resamp, pxxAR_resamp, varphi, varNoise] = getARstats(st, windows, deltaT, quant, order, true, pw_time);
        
         figure;
         subplot(2, 3, 1);
         hold on;
         loglogDist(sizes, 'g', '.', 12, false, false, 0, 0, 1, true);
         loglogDist(sizesAR, 'k', '.', 12, false, false, 0, 0, 1, true);
         legend('Experiment', 'AR Fit');
         xlabel('Avalanche Size, S')
         ylabel('Probability Density, P(S)')
         box off;

         subplot(2, 3, 4);
         hold on;
         loglogDist(durations, 'g', '.', 12, false, false, 0, 0, 1, true);
         loglogDist(durationsAR, 'k', '.', 12, false, false, 0, 0, 1, true);
         xlabel('Avalanche Duration, T')
         ylabel('Probability Density, P(T)')
         box off;

         %power spectrum 
         subplot(2, 3, 2);
         hold on;
         plot(freq, pxx_resamp, '.g', 'MarkerSize', 12);
         plot(freq, pxxAR_resamp, '.k', 'MarkerSize', 12);
         xlabel('Frequency (Hz)')
         ylabel('Power Spectral Density')
         set(gca, 'xscale', 'log', 'yscale', 'log')
         box off;
         
         %DFA
         subplot(2, 3, 5);
         hold on;
         plot(lengths, F, '.g', 'MarkerSize', 12);
         plot(lengths, FAR, '.k', 'MarkerSize', 12);
         xlabel('Window Length w (s)')
         ylabel('F(w)')
         set(gca, 'xscale', 'log', 'yscale', 'log')
         box off;

         subplot(2, 3, 3);
         hold on;
         %simulate AR model in same windows as experimental data
         sigList = simARwindowed(varphi, varNoise, round((windows(:, 2)-windows(:, 1))/deltaT));
         sigAR = [];
         for i = 1:length(sigList)
             curr = sigList{i};
             sigAR = [sigAR ; curr(:)];
         end
         histogram(sigAR, 20, 'FaceColor', 'k', 'FaceAlpha', 0.5, 'LineStyle', 'none');

         sigExpt = zscore(getFR(windows, st, deltaT));
         histogram(sigExpt, 20, 'FaceColor', 'g', 'FaceAlpha', 0.5, 'LineStyle', 'none');
         xlabel('Activity')
         ylabel('Count')
         box off;

         subplot(2, 3, 6);
         plot(beta, d_beta, '.k', 'MarkerSize', 8);
         xlabel('\beta')
         ylabel('d_\beta')
         set(gca, 'yscale', 'log')
         set(gcf, 'Color', 'w')
         box off;
     end
end