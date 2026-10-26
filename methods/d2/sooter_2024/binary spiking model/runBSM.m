function [S] = runBSM(N, pCon, ItoE, lambdaMax, p, T, varFR, loc, l)
if loc
    C = getLocCon(round(sqrt(N)), l);
    N = size(C, 1);
else
    C = double(rand(N, N) < pCon);
end

C(:, round(4*N/5):end) = -ItoE*C(:, round(4*N/5):end);

if varFR
    C = C.*rand(N, 1);
end
C = C*lambdaMax/max(abs(eig(C)));

S = zeros(N, T);
S(:, 1) = rand(N, 1) < 0.1;
for t = 2:T
    input = p + C*S(:, t-1);
    S(:, t) = rand(N, 1) < input;
end
end