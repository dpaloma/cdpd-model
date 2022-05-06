warning('off','all')
%%
% Parámetros
a = 0.05;
lambda_0 = 0.05;
delta = 0.05;
rho = 3;
sigma = 0.1;
alpha = 100;
beta = 50;
anual = 52.14;
maturity = 1/anual;
T = anual*maturity;
%%
%% Calculo del bono la ecuación 26
% En este caso $g$ hace referencia a los saltos internos y $h$ a los saltos
% externos, para la ecuación (26) del artículo, se hace $h=g=1$ para dar a
% entender que se trabajara con ambos tipos de saltos.
%% Obtención de $a^{+}
h = 1;    g = 1;
f = @(x) ecu_a(x,delta,sigma,alpha,g,h);
a_mas = fzero(f,0);
A = G01(a_mas,delta,sigma,alpha,g,h,T);
B26 = B(T,a,lambda_0,delta,sigma,beta,alpha,rho,a_mas,h,g)

plot(x,y);
hold on
plot([a_mas,a_mas],[3,-3]);
hold on
plot([0,3],[0,0]);
%%
%%
%% Formula para $\widehat{g}(u)$ y  $\widehat{h}(u)$
% Formula para $\widehat{g}(u)$, tomandola como la la función generadora de
% momentos. Como en este caso se supone que todo es exponencial 
function gg = g_gorro(u,alpha)
    gg = 1./(1-u/alpha);
end
% Formula para $\widehat{g}(u)$, tomandola como la la función generadora de
% momentos. Como en este caso se supone que todo es exponencial 
function hh = h_gorro(u,beta)
    hh = 1./(1-u/beta);
end
%%
% Ecuación para encontrar $a^{+}$
function res = ecu_a(A,delta,sigma,alpha,g,h)
    res = 2-delta.*A-g*g_gorro(A,alpha)-0.5.*h*sigma^2.*A.^2;
end
%%
% Elaboración del valor $\mathcal{G}^{-1}_{0,1}(T)$
function RA = G01(a_mas,delta,sigma,alpha,g,h,T)
    f = @(x) 1./ecu_a(x,delta,sigma,alpha,g,h);
    g = @(y) integral(f,0,y)-T;
    RA = fzero(@(y) g(y),[0,a_mas]);
end
%%
%% Construcción de la función $B(0,T)$
function R = B(T,a,lambda_0,delta,sigma,beta,alpha,rho,a_mas,h,g)
   G_menos = G01(a_mas,delta,sigma,alpha,g,h,T);
   f1 = @(u) a.*delta.*u+h*rho.*(1-h.*h_gorro(u,beta));
   f2 = @(u) 2-delta.*u-g.*g_gorro(u,alpha)-0.5.*sigma.^2+u.^2;
   q = integral(@(u) f1(u)./f2(u),0,G_menos);
   term = G_menos*lambda_0+q;
   R = exp(-term);
end