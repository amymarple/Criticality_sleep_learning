function [pxx, freq, numUsed] = myPSD_sigList(sigList, deltaT, pw_width)
%pw_width in number of samples
count = 0;
numUsed = 0;

for i = 1:length(sigList)
    sig = sigList{i};
    if length(sig) > pw_width
        [pxxCurr, freqCurr] = pwelch(sig, pw_width, [], [], 1/deltaT);
        numWindows = floor(length(sig)/(pw_width/2));
        if numUsed == 0
            freq = freqCurr;
            pxx = pxxCurr*numWindows;
        else
            pxx = pxx + pxxCurr*numWindows;
        end
        count = count + numWindows;
        numUsed = numUsed + 1;
    end
end
pxx = pxx/count;
end