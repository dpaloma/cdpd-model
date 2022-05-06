delta = 0.05;
beta = 50;
obs = 1000;
% sigmas = [0.001,0.01,0.1,0.3,0.5,1];
sigmas = [0.001,0.1,0.5,1,2,5,10];
[~,cols] = size(sigmas);
a_mas = zeros(size(sigmas));
maxT = zeros(size(sigmas));
a_m = 0;
puntos_graf = zeros([obs, 2*cols]);
g_gorro = @(u) beta./(u+beta);
%%
% Ecuación 11 en terminos de u, $\delta$ y $\sigma$.
ecu_a = @(u,delta,sigma) 2-delta.*u-g_gorro(u)-0.5.*sigma.^2.*u.^2;
% for que administra cada una de las funciones diferentes que se generan
% con cada valor en el vecto sigmas.
for i = 1:length(sigmas)
    %%
    % función en terminos de $u$.
    f = @(u) ecu_a(u,delta,sigmas(i)); 
    % Obtención del valor $a^{+}$
    a_m = fzero(f,[eps,30]);
    a_mas(i) = a_m;
    x = linspace(0,a_m,obs);
    y = zeros(size(x));
    for j = 1:length(x)
        y(j) = f(x(j));
    end
    % Valor máximo para $T$
    f2 = @(x) 1./ecu_a(x,delta,sigmas(i));
    RA = integral(f2,0,a_m);
    maxT(i) = RA;
    puntos_graf(:,[2*i-1,2*i]) = [x',y'];
    if i ~= 1
        hold on
        plot(x,y)
    else
       plot(x,y) 
    end
end
% Generación de la tabla con los datos para los gráficos.
tabla = table(puntos_graf(:,1),puntos_graf(:,2),puntos_graf(:,3),...
    puntos_graf(:,4),puntos_graf(:,5),puntos_graf(:,6),puntos_graf(:,7),...
    puntos_graf(:,8),puntos_graf(:,9),puntos_graf(:,10),puntos_graf(:,11),...
    puntos_graf(:,12));
tabla.Properties.VariableNames = {'x1','s1','x2','s2','x3','s3','x4','s4',...
    'x5','s5','x6','s6'};
writetable(tabla,'graf_a_mas.csv');
% Tabla de los valores de $a^{+}$
relacion = [sigmas',a_mas'];
%%
% Generación de los limites superiores de los intervalos $[0,\mathcal{G}_{0,1}^{-1}(T))$
relacion2 = [sigmas',a_mas',maxT'];