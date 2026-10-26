load('...\example case\YMV17.mat');
NREM = double(NREM);
WAKE = double(WAKE);

deltaT = 0.04;
order = 20;
check_fit = true;
quant = 0.1;
for whichState = [1, 2]

    if whichState == 1
        windows = WAKE;
    else
        windows = NREM;
    end

    [beta, d_beta, closest_models] = getDbetas(st, windows, deltaT, order, check_fit, quant);

    if whichState == 1
        disp(['wake d2 = ', num2str(d_beta(1))])
    else
        disp(['sleep d2 = ', num2str(d_beta(1))])
    end
    
end