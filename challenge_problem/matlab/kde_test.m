%% kde_test
% kernel density estimate test for multimodality of data

clear variables;
close all;

% try to load python file
if exist('../prelim_analysis/prelim.txt','file')
    M = csvread('../prelim_analysis/prelim.txt');
    tspan = M(:,1)';
    XK = M(:,2:end);
else
    load data.mat;
end

Np = size(XK,2)/2;

% number of points to evaluate for unimodality
ntry = 100;
indtry = round(linspace(1,length(tspan),ntry));

hcrit_v = zeros(ntry,1);
P_unimodal = zeros(ntry,1);
%%
tic;
for nouter = 1:ntry

    duse = XK(indtry(nouter),:);

    X1 = duse(1:2:(2*Np-1));% position states
    X2 = duse(2:2:2*Np);% velocity states
    %% binary search to determine the critical value of h: that is, the value for which there is only one local maximum

    % hu: the search variable; smoothing parameter for KDE
    hu = 10.0;

    % grid of points
    hug = [0.1 10.0];

    % number of iterations
    itermax = 50;
    % tolerance required to break iterating
    itertol = 1e-2;

    for iter = 1:itermax

        hu = mean(hug);

        n = Np;
        x1grid = linspace(-15,15,25);
        x2grid = linspace(-10,10,20);

        pdf2d = @(x,mu,P) 1/sqrt(2*pi*det(P))*exp(-1/2*(x-mu)'*(P\(x-mu)));

        kdegrid = zeros(length(x1grid),length(x2grid));
        for k = 1:length(x1grid)
            for j = 1:length(x2grid)
                for h = 1:n
                    kdegrid(k,j) = kdegrid(k,j) + 1/(n*hu) * pdf2d( ([x1grid(k);x2grid(j)]-[X1(h);X2(h)] )/hu,0.0,diag([1.0,1.0]));
                end
            end
        end

        % evaluate the number of local maxima in kdegrid
        %fprintf('h = %f, extrema:\n',hu);
        %fprintf('     j,     k,    x1,    x2,kde(j,k)\n')

        next = 0;
        for k = 2:length(x1grid)-1
            for j = 2:length(x2grid)-1
                if kdegrid(k,j) > kdegrid(k-1,j) && kdegrid(k,j) > kdegrid(k,j-1) && kdegrid(k,j) > kdegrid(k+1,j) && kdegrid(k,j) > kdegrid(k,j+1)
                    next = next+1;
                    %fprintf('%6d,%6d,%6.3f,%6.3f,%8.5f\n',k,j,x1grid(k),x2grid(j),kdegrid(k,j));
                end
            end
        end

        if next == 1
            hug(2) = hu;
        else
            hug(1) = hu;
        end

        if diff(hug) < itertol
            break
        end

    end

    hcrit = hug(2);
    hcrit_v(nouter) = hcrit;
    %% now perform the check for significance
    miter = 100;
    next = zeros(miter,1);

    for mi = 1:miter

        % repeat 'miter' times

        % sample smoothed bootstrap samples from the KDE
        % sample from the KDE with the identified critical value , hu = mean(hug)
        % get n samples from the KDE
        nsamp = round(rand(n,1)*(n-1)+1);
        % bootstrap sample
        XKK = [X1(nsamp)' X2(nsamp)'];
        sigmat = std([X1;X2],1,2);
        % vectorized computation
        e1 =randn(n,2);
        XKK = (XKK+hcrit.*e1)./sqrt(1+hcrit^2./repmat(sigmat.^2',n,1));
        
        %% now evaluate the number of modes with the critical smoothing factor

        hu = hcrit;

        n = Np;
        x1grid = linspace(-15,15,25);
        x2grid = linspace(-10,10,20);

        kdegrid = zeros(length(x1grid),length(x2grid));
        for k = 1:length(x1grid)
            for j = 1:length(x2grid)
                for h = 1:n
                    kdegrid(k,j) = kdegrid(k,j) + 1/(n*hu) * pdf2d( ([x1grid(k);x2grid(j)]-[XKK(h,1);XKK(h,2)] )/hu,0.0,diag([1.0,1.0]));
                end
            end
        end

        % evaluate the number of local maxima in kdegrid
        %fprintf('h = %f, extrema:\n',hu);
        %fprintf('     j,     k,    x1,    x2,kde(j,k)\n')

        next(mi) = 0;
        for k = 2:length(x1grid)-1
            for j = 2:length(x2grid)-1
                if kdegrid(k,j) > kdegrid(k-1,j) && kdegrid(k,j) > kdegrid(k,j-1) && kdegrid(k,j) > kdegrid(k+1,j) && kdegrid(k,j) > kdegrid(k,j+1)
                    next(mi) = next(mi)+1;
                    %fprintf('%6d,%6d,%6.3f,%6.3f,%8.5f\n',k,j,x1grid(k),x2grid(j),kdegrid(k,j));
                end
            end
        end
    end

    P_unimodal(nouter) = length(find(next<=1))/miter;
    fprintf('Approximate significance level: P = %f for 1 mode\n', P_unimodal(nouter));
    etaCalc(nouter,ntry,toc);

end

%% draw the probability and the KDE as a function of time
ANIMATE = 0;
if ANIMATE
    figure('Renderer','zbuffer')
    
    k = 1;
    
    subplot(121);
    plot(XK(1:k,1:2:(2*Np-1)),XK(1:k,2:2:(2*Np)));
    title('Phase portrait');
    set(gca,'NextPlot','replaceChildren','ylim',[-10 10],'xlim',[-20 20]);
    
    subplot(122);
    plot(tspan(indtry(1:k)), P_unimodal(1:k));
    title('Probability of unimodality');
    set(gca,'NextPlot','replaceChildren','ylim',[0.75 1.0],'xlim',[tspan(1) tspan(end)]);
    
    for k = 2:ntry
        subplot(121);
        %plot(XK(indtry(1:k),1:2:(2*Np-1)),XK(indtry(1:k),2:2:(2*Np)));
        plot(XK(indtry(k),1:2:(2*Np-1)),XK(indtry(k),2:2:(2*Np)),'o');
        
        subplot(122);
        plot(tspan(indtry(1:k)), P_unimodal(1:k));
                
        pause(0.1);
    end
end