function [varphi, varNoise] = myYuleWalker(st, windows, deltaT, order)
[lags, acf] = myAutocorr(st, windows, deltaT, order);
orderLagACF = acf(end);
acf = acf(1:(end-1));
if size(acf, 1)<size(acf, 2)
    acf = acf';
end

r = [flip(acf, 1) ; 1 ; acf];
R = zeros(order, order);
for i = 1:order
    idx = 1-i-(1-order)+1;
    R(:, i) = r(idx:(idx+order-1));
end

varphi = inv(R)*[r((order+1):end) ; orderLagACF];

varNoise = 1 - [acf ; orderLagACF]'*varphi;%careful, +1 is not right

sigAll = getFR(windows, st, deltaT);
n = length(sigAll);
end