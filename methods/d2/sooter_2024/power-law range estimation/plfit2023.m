function [tau, bestrho, bestmm, bestMM, nsamp, bestfracgood, continuous] = plfit2023(temp, gof, plotflag)%plfitK7
% avalanche size distribution analysis

% maximum likelihood fit for powerlaw with upper and lower cutoffs

% 'temp' is a vector of event sizes (or durations).  These are discrete,
% but not necessarily integers
% 'plotflag' should be set to 0 if you don't want to generate plots
% (faster)

%goodness of fit (GOF) threshold: we seek the largest PL range that passes this GOF

%make data into sorted column vector if it's not already
temp=sort(temp);
[nr, ~]=size(temp);
if nr==1; temp=temp'; end

nav=length(temp); %number of events in data set

dbr=log10(max(temp))-log10(min(temp));
binv=logspace(log10(min(temp)),log10(max(temp)),10*dbr);
ns=length(binv);

CONTINUOUS = ~(rem(max(temp)-min(temp), 1) == 0);


%make pdf for plot
if plotflag
    plotbins_all=binv(1:end-1)+diff(binv)/2;
    n=histc(temp,binv)'; n(end)=[];
    pdf_all=n./diff(binv)/nav;

    plotbins=binv(1:end-1)+diff(binv)/2;
end

%list of power law exponents to try fitting
exlist=0.5:0.02:5;
nex=length(exlist);

bestMM=0; %upper end of powerlaw range
bestmm=0; %lower end of powerlaw range
tau=0;  %exponent
bestrho=0; %powerlaw range
nsamp=0;  %number of samples within the powerlaw range
bestfracgood=0; %GOF for the best case
continuous = CONTINUOUS;


for mmm=1:ns-1 %loop through different lower cutoffs
    for MMM = (mmm + 1):ns
        %truncate the data set
        mm=binv(mmm); %lower end of range
        MM=binv(MMM); %upper end of range
        z=temp(temp>=mm & temp<=MM);
        nz=length(z);

        %compute range of truncated data set
        if CONTINUOUS
            rho = log10(MM/mm);
        else
            rho=log10(max(z)/min(z));
        end

        if nz>20 && (min(z) ~= max(z)) && rho>bestrho
            %list of discrete values of the data between the lower and
            %upper cutoffs
            xx = min(z):max(z);
            %find max likelihood power-law exponent
            L = zeros(1,nex);
            for q=1:nex
                aa=exlist(q); %exponent
                if CONTINUOUS
                    normf = (MM^(1-aa)-mm^(1-aa))/(1-aa);
                else
                    normf = sum(xx.^(-aa));
                end
                prs=(1/normf)*z.^-aa; %probabilities
                L(q)=mean(log(prs)); %log likelihoods
            end
            [~,ind]=max(L); %find max likelihood exponent
            aa=exlist(ind);

            %resampled data CDF: 10 points per decade logarithmically spaced
            rz=logspace(log10(min(z)),log10(max(z)),10*rho); 
            cprob = sum(z<=rz)/nz;

            %calculate theoretical CDF at the logarithmically spaced points rz
            theCDF = zeros(1, length(rz));
            if CONTINUOUS
                theCDF = (rz.^(1-aa) - mm^(1-aa))/(MM^(1-aa) - mm^(1-aa));
            else
                for i = 1:length(rz)
                    theCDF(i) = 1/sum(xx.^(-aa))*sum(xx(xx<=rz(i)).^(-aa));
                end
            end

            vbounds = zeros(2, length(rz));
            leeway = 0.03; %caution: I changed this (was 0.03)
            vbounds(1, :) = theCDF - leeway;
            vbounds(2, :) = theCDF + leeway;

            fracgood=mean(cprob>(vbounds(1,:)) & cprob<(vbounds(2,:)));
           
            if  fracgood>gof
                bestMM=MM;
                bestmm=mm;
                tau=aa;
                bestrho=rho;
                nsamp=nz;
                bestfracgood=fracgood;

                disp(['rho=',num2str(rho), 'aa=',num2str(aa),'fg=',num2str(fracgood),'nsamp=',num2str(nsamp)])

                %show PDF and CDF plots
                if plotflag
                    figure(100)

                    %PDF
                    subplot(121)
                    loglog(plotbins_all, pdf_all,'.k'); %all data, no outliers excluded
                    hold on;

                    n=histc(temp,binv)'; n(end)=[];
                    loglog(plotbins, n./diff(binv)/nz,'go'); %all data except outliers

                    n2=histc(z,binv)';
                    n2(end)=[];
                    loglog(plotbins, n2./diff(binv)/nz,'r'); %only the data within pl range

                    loglog(plotbins(mmm:(MMM-1)),(aa-1)/(mm^(1-aa)-MM^(1-aa))*plotbins(mmm:(MMM-1)).^-aa,'r') %best fit pl
                    hold off;

                    %CDF
                    subplot(122)
                    semilogx(rz,vbounds(1,:),'.m') %min bound for surrogates
                    hold on
                    semilogx(rz,vbounds(2,:),'.m') %max bound for surrogates
                    semilogx(z,(1:nz)/nz,'r.') %data CDF
                    semilogx(rz,cprob,'b+') %resampled data CDF
                    hold off
                end
            end
        end
    end
end
end
