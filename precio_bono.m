warning('off','all')
%% Precio del bono y termino $a^{+}$
% Parámetros
global beta alpha;
a = 0.05;
lambda_0 = 0.05;
delta = 0.05;
rho = 3;
sigma = 0.9;
alpha = 100;
beta = 50;
T = 12;
%%
% h saltos externos
% g saltos internos
h = 1;    g = 1;
%%
% Funciones $\widehat{h}$ y $\widehat{g}$
h_gorro = @(u) h*alpha./(u+alpha);

g_gorro = @(u) g*beta./(u+beta);
%%
% Ecuación 11
ecu_a = @(A,delta,sigma) 2-delta.*A-g_gorro(A)-0.5.*sigma.^2.*A.^2;

f1 = @(x) ecu_a(x,delta,sigma);

options = optimset('TolFun',1e-12);
% Calculo del término $a^{+}$
% a_mas = fminbnd(f1,eps,10^6,options);
a_mas = fzero(f1,[eps,30]);
%%
f2 = @(x) 1./ecu_a(x,delta,sigma);
%%
% busqueda del termino $T$ talque $\mathcal{G}_{0,1}(y) = T$
gg2 = @(y) integral(f2,0,y);
g2 = @(y) abs(integral(f2,0,y)-T);
RA = fminbnd(g2,eps,a_mas,options);

x = linspace(0,a_mas,1000);
y = zeros(size(x));
for i = 1:length(x)
   y(i) = gg2(x(i)); 
end
plot(x,y)
%%
%% Construcción de la función $B(0,T)$
ff1 = @(u) a.*delta.*u+h*rho.*(1-h_gorro(u));
q =@(z) integral(@(u) ff1(u)./ecu_a(u,delta,sigma),0,z);
term = -RA*lambda_0-q(RA);
R = exp(term);
% Precio final del bono y valor de $a^{+}
[R a_mas]

% %%
% %%  Recreación de las tablas del artículo
% % Recreación tabla 5.
% bonohg = @(h,g) f_precio_bono(a,lambda_0,delta,rho,sigma,alpha, beta,h,g,T);
% casos = categorical({'Todos los saltos';'Solo saltos internos';'Solo datos externos'});
% H = [1,1,0];
% G = [1,0,1];
% v_bonos = [bonohg(H(1),G(1)),bonohg(H(2),G(2)),bonohg(H(3),G(3))];
% table(casos,H',G',v_bonos','VariableNames',{'Caso','Salto Externo (h)','Salto Interno (g)','B(0,1)'})
% %%
% % Recreación tabla 6.
% v_sigma = [0.01;0.1;0.5;0.8;10];
% B_sigma = zeros(size(v_sigma));
% bonosigma = @(x) f_precio_bono(a,lambda_0,delta,rho,x,alpha, beta,h,g,T);
% for bs = 1:max(size(v_sigma))
%     B_sigma(bs) = bonosigma(v_sigma(bs));
% end
% table(v_sigma,B_sigma,'VariableNames',{'sigma','B(0,1)'})
% %%
% % Gráfico del precio del bono respecto a $\sigma$.
% h = 1;   g = 1;
% g = @(X) f_precio_bono(a,lambda_0,delta,rho,X,alpha, beta,h,g,T);
% %%
% % Cosntrucción del gráfico
% x = linspace(0,50,100);
% y = zeros(size(x));
% for i = 1:length(x)
%     y(i) = 100*g(x(i));
% end
% 
% plot(x,y)

%%
%%
%%
%%
% Alcance
sigmas = [0.00001,0.0001,0.001,0.01,0.1,0.3,0.5,0.8,0.9];
b_mas = zeros(size(sigmas));
T_max = b_mas;

for i = 1:length(b_mas)
   f2 = @(x) ecu_a(x,delta,sigmas(i));
   
   b_mas(i) = fzero(f2,[eps,30]);
   
   % Función $\mathcal{G}_{0,1}(y)$
   ff2 = @(y) 1./f2(y);
   GUC = @(y) integral(ff2,0,y);
   T_max(i) = GUC(b_mas(i));
end

x = linspace(0,b_mas(2),1000);
y = zeros(size(x));
for i = 1:length(x)
    y(i) = 100*g2(x(2));
end