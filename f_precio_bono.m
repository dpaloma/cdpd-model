function [R,a_mas] = f_precio_bono(a,lambda_0,delta,rho,sigma,alpha, beta,h,g,T)
    h_gorro = @(u) h*alpha./(u+alpha);

    g_gorro = @(u) g*beta./(u+beta);
    %%
    % Ecuación 11
    ecu_a = @(A,delta,sigma) 2-delta.*A-g_gorro(A)-0.5.*sigma.^2.*A.^2;

    f1 = @(x) ecu_a(x,delta,sigma);

    options = optimset('TolFun',1e-12);
    
    a_mas = fzero(f1,[eps,30]);
    %%
    f2 = @(x) 1./ecu_a(x,delta,sigma);
    g2 = @(y) abs(integral(f2,0,y)-T);
    RA = fminbnd(g2,eps,a_mas,options);

    %% Construcción de la función $B(0,T)$
    ff1 = @(u) a.*delta.*u+h*rho.*(1-h_gorro(u));
    q =@(z) integral(@(u) ff1(u)./ecu_a(u,delta,sigma),0,z);
    term = -RA*lambda_0-q(RA);
    R = exp(term);
end