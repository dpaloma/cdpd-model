%% Paramétros
a = 0.05;
lambda_0 = 0.05;
delta = 0.05;
rho = 3;
% sigma = 0.1;
alpha = 100;
beta = 50;
anual = 52.14;
maturity = 1/anual;
T = anual*maturity;
% Con ambos tipos de saltos
h = 1;   g = 1;
%%
% Precio del bono como funcion de la volatilidad $\sigma$.

b = @(X) f_precio_bono(a,lambda_0,delta,rho,X,alpha, beta,h,g,T);

% Dominio de la función
x = linspace(0,10,1000);

% Construcción de la imagen
y = zeros(size(x));
for i = 1:length(x)
    y(i) = 100*b(x(i));
end

%plot(x,y)

%%
% Precio del bono cero cupon para $t=0$ y distintos sigma

sigmas = [0.001,0.1,0.3,0.4,0.5,1];

% El dominio es el mismo que se definio para el gráfico anterior.
% Las imagenes se guardan en una sola matriz.
imagenes = [];

for i = 1:length(sigmas)
    y = zeros(size(x));
    bb = @(X) f_precio_bono(a,lambda_0,delta,rho,sigmas(i),alpha, beta,h,g,X);
    for j = 1:length(x)
        y(j) = 100*bb(x(j));
    end
    imagenes = [imagenes,y'];
end

grafico = [x',imagenes];

plot(grafico(:,1),grafico(:,2:end))

writematrix(grafico,'graf_sigmas.csv');