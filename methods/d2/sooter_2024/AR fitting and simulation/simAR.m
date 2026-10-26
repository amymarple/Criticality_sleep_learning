function [sig] = simAR(varphi, varNoise, T)
    %make model capsules
    model = arima(length(varphi), 0, 0);
    model.Constant = 0;
    model.Variance = varNoise;
    model.AR = num2cell(varphi);

    %simulate
    sig = simulate(model, T);
end