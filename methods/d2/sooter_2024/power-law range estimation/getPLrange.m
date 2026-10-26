function [xMin, xMax, range, exponent] = getPLrange(x, y, whichWay, expReq, gof, leeway)

y = y(x>0);
x = x(x>0);
if size(x, 2) > size(x, 1)
    x = x';
end

if size(y, 2) > size(y, 1)
    y = y';
end

xMin = 0;
xMax = 0;
range = 0;
exponent = 0;

tryThese = logspace(log10(x(1)), log10(x(end)), round(10*log10(x(end)/x(1))));
for i = tryThese
    for j = tryThese(tryThese>i)
        disp([i/max(tryThese), j/max(tryThese)])
        if log10(j/i) > range
            xx = x(i<=x & x<j);
            yy = y(i<=x & x<j);

            [abc, S] = polyfit(log10(xx), log10(yy), 1);
            pred = log10(xx)*abc(1) + abc(2);
            error = abs(log10(yy)-pred);

            frac_good = sum(1*(error<leeway))/length(error);
            exp = abc(1); 
            if whichWay == 'g'
                cond = exp>expReq;
            else
                cond = exp<expReq;
            end

            if frac_good>gof & cond 
                xMin = i;
                xMax = j;
                range = log10(j/i);
                exponent = exp; 
            end
        end
    end
end
end