function [Av] = getAvShapes(windows, quant, deltaT, spks, isiBin, shapesToo, zscoreSig, phase_shuffle)

if isiBin
    thresh = 0;
else
    if size(spks, 1) > size(spks, 2)
        spks =spks';
    end
    sigAll = [];
    for i = 1:size(windows, 1)
        sigAll = cat(2, sigAll, histc(spks, windows(i, 1):deltaT:windows(i, 2)));
    end
    
    thresh = quantile(sigAll, quant);
    if zscoreSig
        thresh = quantile(zscore(sigAll), quant);
    end
end

Av.duration = [];

Av.size = [];

if shapesToo
    Av.shape = {};
end
AvCounter = 1;
for i = 1:size(windows, 1)
    if isiBin
        deltaT = getISI([[windows(i, 1) windows(i, 2)]], spks);
    end
    sig = histc(spks, windows(i, 1):deltaT:windows(i, 2));
    if zscoreSig
        sig = (sig-mean(sigAll))/std(sigAll);
    end

    if phase_shuffle
        sig = phaseShuffle(sig);
    end
    
    below = find(sig <= thresh);
    for j = 1:(length(below) - 1)
        lower = below(j);
        upper = below(j + 1);
        if (upper - lower) > 1
            currProf = sig((lower + 1):(upper - 1)) - thresh;
            currSize = sum(currProf);
            if isiBin
                currDur = deltaT*(upper - lower - 1);
            else
                currDur = upper - lower - 1;
            end
            Av.duration = [Av.duration currDur];
            Av.size = [Av.size currSize];
            if shapesToo
                Av.shape{AvCounter} = currProf;
            end
            AvCounter = AvCounter + 1;
            
        end
    end
end

end