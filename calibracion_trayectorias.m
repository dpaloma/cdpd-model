%% Exploración de los datos del banco de la republica.
%
%% Extracción del conjunto de entrenamiento para la calibración de los datos.
%
%%
% Historial de datos original.
warning('off','all')
H = readmatrix('tes_corto_plazo.xlsx','Range','A2:H323');
H = H(:,[1,2,7,8]);
%%
% Modificación del tiempo
ag_time = 693960;
H(:,2) = H(:,2)+ ag_time;
%%
% % Calculo de las diferencias
% dif = [H(1:end-1,5)-H(2:end,5);0];
% %%
% % se agreaga a la base original
% H = [H(:,1:5) dif];
%%
% La base tiene las tasas para 1 año, 5 años y 10 años. En este caso nos
% quedamos solamente con la fecha y la tasa a un año.
% Historial de los datos entre 2017 y 2020, tomando también el último dato
% del 2016
ind = logical((H(:,1)==2017) + (H(:,1)==2018) + (H(:,1)==2019) + (H(:,1)==2020)+(H(:,1)==2021)+(H(:,1)==2022));
% ind(max(t)+1) = true;
H = H(ind,:);
[~, I] = sort(H(:,2));
H = H(I,:);
%%
% Asignacion tiempos
t = H(:,2);
%% Gráfico del historial de datos
%
plot(t,H(:,3));
title('Historial de la tasa semanal 1y entre 2017 y 2020'); 
ylabel('Porcentaje (%)'); 
xlim([min(t),max(t)]);
datetick('x',11,'keeplimits');
%%
%% Calibración de parámetros por trimestre
%
% Fechas del primer viernes de cada trimestre entre 2017 y el primer
% trimestre de 2022.
% Fechas de inicio para cada trimestre
start_year = [2017+zeros(1,4),2018+zeros(1,4),2019+zeros(1,4),2020+zeros(1,4),2021+zeros(1,4),2022]';
start_mounth = [1;4;7;10;1;4;7;10;1;4;7;10;1;4;7;10;1;4;7;10;1];
start_day = [3;4;4;3;2;3;3;2;2;2;2;1;7;7;7;6;5;6;6;5;4];
start_date = [start_year,start_mounth,start_day];
% Fechas de finales para cada trimestre
end_mounth = [1;4;7;10;1;4;7;10;1;4;7;10;1;4;7;10;1;4;7;10;1]+2;
end_day = [28;27;26;26;7;26;25;26;26;25;24;17;31;30;29;15;30;29;28;14;29];
end_date = [start_year,end_mounth,end_day];
num_fechas = [datenum(start_date),datenum(end_date)];
num_fechas = [num_fechas,zeros(size(num_fechas(:,1:2)))];
obs = zeros(6,2);
A = unique(H(:,1));
for i = 1:6
    if i ~= 6
        obs(i,:) = [A(i),sum(H(:,1) == A(i))];
    else
        caso2022 = zeros([52,1]);
        j = 1;
        caso2022(j) = 738525;
        j = 2;
        while caso2022(j-1) < 738868
            caso2022(j) = caso2022(j-1)+7;
            j = j+1;
        end
        caso2022 = caso2022(caso2022 ~=0);
        obs(i,:) = [2022,sum(caso2022 ~= 0)];
    end
end

for i = 0:4
   num_fechas(4*i+(1:4),end-1:end) = [obs(i+1,:);obs(i+1,:);obs(i+1,:);obs(i+1,:)]; 
end
num_fechas(end,end-1:end) = obs(end,:);
% Conjuntos de parámetros calibrados
c_a = zeros(21,1);  c_lambda_0 = zeros(21,1);
c_delta = zeros(21,1);  c_sigma = zeros(21,1);
c_beta = zeros(21,1);   c_alpha = zeros(21,1);
c_rho = zeros(21,1);
%
% Busqueda de los parámetros via optimización
% parámetros iniciales para los procesos de optimización
a = 0.9;
delta = 1;
sigma = 0.8;
beta = 1.2;
alpha = 10;
rho = 0.1;
x0 = [a,delta,sigma,beta,alpha,rho];
% Calibración de los parámetros para cada uno de los trimestre
for i = 1:13
    disp(i)
    F1 = num_fechas(i,1);
    F2 = num_fechas(i,2);
    HF1 = H(:,2)>=F1;   HF2 = H(:,2)<=F2;
    HT = H(logical(HF1.*HF2),3);
    RT = exp(-HT/100);
    lambda_0 = HT(1);

    options = optimset('MaxFunEvals',700,'TolFun',1e-5,'MaxIter',10000);
    % options = optimset('TolFun',1e-5);
    % options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt',...
    %     'MaxFunctionEvaluations',1500)

    fun = @(x) calibracion(52,lambda_0,x(1),x(2),x(3),x(4),x(5),x(6),RT);

    % [x,fval] = fminsearch(fun,x0,options)
    goal = 0;
    weight = 0;
    lb = eps+zeros(size(x0));
    ub = 100+zeros(size(x0));
    xx = fgoalattain(fun,x0,goal,weight,[],[],[],[],lb,ub,[],options);
    c_a(i) = xx(1);  c_lambda_0(i) = lambda_0;
    c_delta(i) = xx(2);  c_sigma(i) = xx(3);
    c_beta(i) = xx(4);   c_alpha(i) = xx(5);
    c_rho(i) = xx(6);
%     x0 =xx;
end

for i = 14:15
    disp(i)
    F1 = num_fechas(i,1);
    F2 = num_fechas(i,2);
    HF1 = H(:,2)>=F1;   HF2 = H(:,2)<=F2;
    HT = H(logical(HF1.*HF2),3);
    RT = exp(-HT/100);
    lambda_0 = HT(1);

    options = optimset('MaxFunEvals',700,'TolFun',1e-5,'MaxIter',10000);
    % options = optimset('TolFun',1e-5);
    % options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt',...
    %     'MaxFunctionEvaluations',1500)

    fun = @(x) calibracion(52,lambda_0,x(1),x(2),x(3),x(4),x(5),x(6),RT);

    % [x,fval] = fminsearch(fun,x0,options)
    goal = 0;
    weight = 0;
    lb = eps+zeros(size(x0));
    ub = 1000+zeros(size(x0));
    xx = fgoalattain(fun,x0,goal,weight,[],[],[],[],lb,ub,[],options);
    c_a(i) = xx(1);  c_lambda_0(i) = lambda_0;
    c_delta(i) = xx(2);  c_sigma(i) = xx(3);
    c_beta(i) = xx(4);   c_alpha(i) = xx(5);
    c_rho(i) = xx(6);
%     x0 =xx;
end

for i = 16:21
    disp(i)
    F1 = num_fechas(i,1);
    F2 = num_fechas(i,2);
    HF1 = H(:,2)>=F1;   HF2 = H(:,2)<=F2;
    HT = H(logical(HF1.*HF2),3);
    RT = exp(-HT/100);
    lambda_0 = HT(1);

    options = optimset('MaxFunEvals',700,'TolFun',1e-5,'MaxIter',10000);
    % options = optimset('TolFun',1e-5);
    % options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt',...
    %     'MaxFunctionEvaluations',1500)

    fun = @(x) calibracion(52,lambda_0,x(1),x(2),x(3),x(4),x(5),x(6),RT);

    % [x,fval] = fminsearch(fun,x0,options)
    goal = 0;
    weight = 0;
    lb = eps+zeros(size(x0));
    ub = 100+zeros(size(x0));
    xx = fgoalattain(fun,x0,goal,weight,[],[],[],[],lb,ub,[],options);
    c_a(i) = xx(1);  c_lambda_0(i) = lambda_0;
    c_delta(i) = xx(2);  c_sigma(i) = xx(3);
    c_beta(i) = xx(4);   c_alpha(i) = xx(5);
    c_rho(i) = xx(6);
%     x0 =xx;
end
P = [c_a, c_lambda_0, c_delta, c_sigma, c_beta, c_alpha, c_rho];

ZZ = esp_lambda(1/52,xx(1),lambda_0/100,xx(2),xx(3),xx(4),xx(5),xx(6))
Z = esp_lambda(2/52,xx(1),lambda_0/100,xx(2),xx(3),xx(4),xx(5),xx(6))
% 
parametros = [xx(1),xx(2),xx(3),xx(4),xx(5),xx(6)];


[A,B] = calibracion(52,lambda_0,xx(1),xx(2),xx(3),xx(4),xx(5),xx(6),RT);

% A = [2017,01,01];    AA = [2017,12,31];
% A = datetime(A);    AA = datetime(AA);
% dA = datenum(2017,01,01);    dAA = datenum(2017,12,31);
% 
% [~, I2017] = sort(H((H(:,1)==2017),1));
% Z = min(H((H(:,1)==2017),2));
% ZZ = recuperar_fecha(Z);
z = datenum(ZZ);
%%
% Gráficos de las traytectorias utilizando parámetros

% Trayectoria por para cada uno de los trimestres
%%
% 2017 trimestre 1
global T1_2017
k = 1;
X = P(k,:); FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2017,T1_2017] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT1_2017 = zeros(size(t_2017)+[15,1]);
TT1_2017(j,:) = [0,t_2017];
TT1_2017(j+1,:) = [lambda_0,T1_2017];
for j = 2:15
    [~,T1_2017] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT1_2017(j+1,:) = [lambda_0,T1_2017];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT1_2017 = [TO';TT1_2017];
%
% 2017 trimestre 2
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2017,T1_2017] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT2_2017 = zeros(size(t_2017)+[15,1]);
TT2_2017(j,:) = [0,t_2017];
TT2_2017(j+1,:) = [lambda_0,T1_2017];
for j = 2:15
    [~,T1_2017] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT2_2017(j+1,:) = [lambda_0,T1_2017];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT2_2017 = [TO';TT2_2017];
%
% 2017 trimestre 3
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2017,T1_2017] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT3_2017 = zeros(size(t_2017)+[15,1]);
TT3_2017(j,:) = [0,t_2017];
TT3_2017(j+1,:) = [lambda_0,T1_2017];
for j = 2:15
    [~,T1_2017] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT3_2017(j+1,:) = [lambda_0,T1_2017];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT3_2017 = [TO';TT3_2017];
%
% 2017 trimestre 4
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2017,T1_2017] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT4_2017 = zeros(size(t_2017)+[15,1]);
TT4_2017(j,:) = [0,t_2017];
TT4_2017(j+1,:) = [lambda_0,T1_2017];
for j = 2:15
    [~,T1_2017] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT4_2017(j+1,:) = [lambda_0,T1_2017];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT4_2017 = [TO';TT4_2017];
%%
% 2018 trimestre 1
global T1_2018
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2018,T1_2018] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT1_2018 = zeros(size(t_2018)+[15,1]);
TT1_2018(j,:) = [0,t_2018];
TT1_2018(j+1,:) = [lambda_0,T1_2018];
for j = 2:15
    [~,T1_2018] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT1_2018(j+1,:) = [lambda_0,T1_2018];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT1_2018 = [TO';TT1_2018];
%
% 2018 trimestre 2
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2018,T1_2018] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT2_2018 = zeros(size(t_2018)+[15,1]);
TT2_2018(j,:) = [0,t_2018];
TT2_2018(j+1,:) = [lambda_0,T1_2018];
for j = 2:15
    [~,T1_2018] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT2_2018(j+1,:) = [lambda_0,T1_2018];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT2_2018 = [TO';TT2_2018];
%
% 2018 trimestre 3
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2018,T1_2018] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT3_2018 = zeros(size(t_2018)+[15,1]);
TT3_2018(j,:) = [0,t_2018];
TT3_2018(j+1,:) = [lambda_0,T1_2018];
for j = 2:15
    [~,T1_2018] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT3_2018(j+1,:) = [lambda_0,T1_2018];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT3_2018 = [TO';TT3_2018];
% 2018 trimestre 4
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2018,T1_2018] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT4_2018 = zeros(size(t_2018)+[15,1]);
TT4_2018(j,:) = [0,t_2018];
TT4_2018(j+1,:) = [lambda_0,T1_2018];
for j = 2:15
    [~,T1_2018] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT4_2018(j+1,:) = [lambda_0,T1_2018];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT4_2018 = [TO';TT4_2018];

%%
% 2019 trimestre 1
global T1_2019
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2019,T1_2019] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT1_2019 = zeros(size(t_2019)+[15,1]);
TT1_2019(j,:) = [0,t_2019];
TT1_2019(j+1,:) = [lambda_0,T1_2019];
for j = 2:15
    [~,T1_2019] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT1_2019(j+1,:) = [lambda_0,T1_2019];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT1_2019 = [TO';TT1_2019];
%
% 2019 trimestre 2
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2019,T1_2019] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT2_2019 = zeros(size(t_2019)+[15,1]);
TT2_2019(j,:) = [0,t_2019];
TT2_2019(j+1,:) = [lambda_0,T1_2019];
for j = 2:15
    [~,T1_2019] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT2_2019(j+1,:) = [lambda_0,T1_2019];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT2_2019 = [TO';TT2_2019];
%
% 2019 trimestre 3
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2019,T1_2019] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT3_2019 = zeros(size(t_2019)+[15,1]);
TT3_2019(j,:) = [0,t_2019];
TT3_2019(j+1,:) = [lambda_0,T1_2019];
for j = 2:15
    [~,T1_2019] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT3_2019(j+1,:) = [lambda_0,T1_2019];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT3_2019 = [TO';TT3_2019];
%
% 2019 trimestre 4
k = 4;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2019,T1_2019] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT4_2019 = zeros(size(t_2019)+[15,1]);
TT4_2019(j,:) = [0,t_2019];
TT4_2019(j+1,:) = [lambda_0,T1_2019];
for j = 2:15
    [~,T1_2019] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT4_2019(j+1,:) = [lambda_0,T1_2019];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT4_2019 = [TO';TT4_2019];
%%
% 2020 trimestre 1
global T1_2020
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2020,T1_2020] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT1_2020 = zeros(size(t_2020)+[15,1]);
TT1_2020(j,:) = [0,t_2020];
TT1_2020(j+1,:) = [lambda_0,T1_2020];
for j = 2:15
    [~,T1_2020] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT1_2020(j+1,:) = [lambda_0,T1_2020];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT1_2020 = [TO';TT1_2020];
%
% 2020 trimestre 2
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2020,T1_2020] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT2_2020 = zeros(size(t_2020)+[15,1]);
TT2_2020(j,:) = [0,t_2020];
TT2_2020(j+1,:) = [lambda_0,T1_2020];
for j = 2:15
    [~,T1_2020] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT2_2020(j+1,:) = [lambda_0,T1_2020];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT2_2020 = [TO';TT2_2020];
%
% 2020 trimestre 3
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2020,T1_2020] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT3_2020 = zeros(size(t_2020)+[15,1]);
TT3_2020(j,:) = [0,t_2020];
TT3_2020(j+1,:) = [lambda_0,T1_2020];
for j = 2:15
    [~,T1_2020] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT3_2020(j+1,:) = [lambda_0,T1_2020];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT3_2020 = [TO';TT3_2020];
%
% 2020 trimestre 4
k = 4;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2020,T1_2020] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT4_2020 = zeros(size(t_2020)+[15,1]);
TT4_2020(j,:) = [0,t_2020];
TT4_2020(j+1,:) = [lambda_0,T1_2020];
for j = 2:15
    [~,T1_2020] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT4_2020(j+1,:) = [lambda_0,T1_2020];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT4_2020 = [TO';TT4_2020];

%%
% 2021 trimestre 1
global T1_2021
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2021,T1_2021] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT1_2021 = zeros(size(t_2021)+[15,1]);
TT1_2021(j,:) = [0,t_2021];
TT1_2021(j+1,:) = [lambda_0,T1_2021];
for j = 2:15
    [~,T1_2021] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT1_2021(j+1,:) = [lambda_0,T1_2021];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT1_2021 = [TO';TT1_2021];
%
% 2021 trimestre 2
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2021,T1_2021] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT2_2021 = zeros(size(t_2021)+[15,1]);
TT2_2021(j,:) = [0,t_2021];
TT2_2021(j+1,:) = [lambda_0,T1_2021];
for j = 2:15
    [~,T1_2021] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT2_2021(j+1,:) = [lambda_0,T1_2021];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT2_2021 = [TO';TT2_2021];
%
% 2021 trimestre 3
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2021,T1_2021] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT3_2021 = zeros(size(t_2021)+[15,1]);
TT3_2021(j,:) = [0,t_2021];
TT3_2021(j+1,:) = [lambda_0,T1_2021];
for j = 2:15
    [~,T1_2021] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT3_2021(j+1,:) = [lambda_0,T1_2021];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT3_2021 = [TO';TT3_2021];
%
% 2021 trimestre 4
k = 4;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2021,T1_2021] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT4_2021 = zeros(size(t_2021)+[15,1]);
TT4_2021(j,:) = [0,t_2021];
TT4_2021(j+1,:) = [lambda_0,T1_2021];
for j = 2:15
    [~,T1_2021] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT4_2021(j+1,:) = [lambda_0,T1_2021];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT4_2021 = [TO';TT4_2021];
%%
% 2022 trimestre 1
k = k+1;
X = P(k,:);FS = num_fechas(k,1);	FE = num_fechas(k,2);
lambda_0 = H(H(:,2)==FS,end-1);
j = 1;
[t_2022,T1_2022] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
TT4_2022 = zeros(size(t_2022)+[15,1]);
TT4_2022(j,:) = [0,t_2022];
TT4_2022(j+1,:) = [lambda_0,T1_2022];
for j = 2:15
    [~,T1_2022] = trayectoria(X,lambda_0,H,num_fechas,FS,FE);
    TT4_2022(j+1,:) = [lambda_0,T1_2022];
end
TO = H(logical((H(:,2)>=FS).*(H(:,2)<=FE)),3);
TT4_2022 = [TO';TT4_2022];

%
%% Escritura de las tablas en .csv
% Primeros trimestres
writematrix(TT1_2017','2017_T1.csv');
writematrix(TT1_2018','2018_T1.csv');
writematrix(TT1_2019','2019_T1.csv');
writematrix(TT1_2020','2020_T1.csv');
writematrix(TT1_2021','2021_T1.csv');
writematrix(TT1_2022','2022_T1.csv');
% Segundo trimestres
writematrix(TT2_2017','2017_T2.csv');
writematrix(TT2_2018','2018_T2.csv');
writematrix(TT2_2019','2019_T2.csv');
writematrix(TT2_2020','2020_T2.csv');
writematrix(TT2_2021','2021_T2.csv');
% Tercer trimestres
writematrix(TT3_2017','2017_T3.csv');
writematrix(TT3_2018','2018_T3.csv');
writematrix(TT3_2019','2019_T3.csv');
writematrix(TT3_2020','2020_T3.csv');
writematrix(TT3_2021','2021_T3.csv');
% Cuartos trimestres
writematrix(TT4_2017','2017_T4.csv');
writematrix(TT4_2018','2018_T4.csv');
writematrix(TT4_2019','2019_T4.csv');
writematrix(TT4_2020','2020_T4.csv');
writematrix(TT4_2021','2021_T4.csv');
%%
%%
%% FUNCIONES
%%
%%
% Recuperar fecha
function date = recuperar_fecha(T)
    date = datetime(T-693960,'ConvertFrom','excel');
end
%% Funciones para el calculo de $\mathbb{E}\left[e^{-v\lambdaT}|\lambda_{0}\right]$
% Formulas para $\widehat{g}(u)$ y $\widehat{h}(u)$
function gg = g_gorro(u,alpha)
    gg = 1./(1-u/alpha);
end
%
function hh = h_gorro(u,beta)
    hh = 1./(1-u/beta);
end
% Númerador
function res = ecu_e(u,delta,sigma,alpha)
    res = delta.*u+g_gorro(u,alpha)-1+0.5*sigma^2.*u.^2;
end
% Elaboración del valor $\mathcal{G}^{-1}_{\nu,1}(T)$
% Para efectos del Teorema 1.3.1 se tiene $\nu=1$
function L = G11(delta,sigma,alpha,T)
    f = @(x) 1./ecu_e(x,delta,sigma,alpha);
    g = @(y) abs(integral(f,y,1)-T);
    options = optimset('TolFun',1e-12);
    L = fminbnd(g,0,1,options);
end
% Valor esperado de $\lambda_{t}$
function lambda = esp_lambda(T,a,lambda_0,delta,sigma,beta,alpha,rho)
   G_menos = G11(delta,sigma,alpha,T);
   f1 = @(u) a.*delta.*u+rho.*(1-h_gorro(u,beta));
   f2 = @(u) ecu_e(u,delta,sigma,alpha);
   q = integral(@(u) f1(u)./f2(u),G_menos,1);
   term = G_menos*lambda_0+q;
   lambda = exp(-term);
end
% Función para la calibración
function [MSE,esp] = calibracion(T,lambda_0,a,delta,sigma,beta,alpha,rho,H)
    Z = size(H);
    Z = Z(1);
    esp = zeros(Z,1);
    t = 1/T;
    for i = 2:Z
        esp(i) = esp_lambda(t*(i-1),a,lambda_0/100,delta,sigma,beta,alpha,rho);
    end
    MSE = sum((H(2:end)-esp(2:end)).^2);
end
% Función generadora de trayectorias
function [t_obs,trayec,dt,lt] = trayectoria(xx,lambda_0,H,num_fechas,FS,FE)
    a =  xx(1);	delta = xx(2);
    sigma = xx(3);	beta = xx(4);
    alpha = xx(5);  rho = xx(6);
    maturity = 1;   anual = 1;
    global E_nm1 As N lambda T ts dWs;
    kappa = sqrt(delta^2+2*sigma^2);
    D = 2*a*delta/(sigma^2);
    %% Funciones de s
    As = @(s) 2*kappa*exp(0.5*s*(kappa+delta));
    Bs = @(s) (exp(kappa*s)-1)*sigma^2;
    Cs = @(s) kappa-delta+(kappa+delta)*exp(kappa*s);
    Es = @(s) kappa+delta+(kappa-delta)*exp(kappa*s);
    Fs = @(s) 2*(exp(kappa*s)-1);
    % Valores iniciales
    N = [];
    T = [];
    lambda = [];
    saltos = [];
    %%
    i=1;
    N(i) = 0;
    T(i) = 0;
    lambda(i) = lambda_0;
    S_nm1 = 0;
    E_nm1 = 0;
    saltos(1:3,i) = [0;0;0];
    while max(T) < maturity*anual
        i = i+1;
        N(i) = 0;
        T(i) = 0;
        lambda(i)=0;
       %% Paso 1.
        % Simulación de $S_{i+1}^{*}$
        %%
        %Lema 3.2
        %%
        saltos(1:3,i) = zeros(3,1);
        compara = false;
        %c = 1;
        while(compara==false)
    %        disp(c)
           U32 = rand(1,3);
            % Paso 1.
            W_g = 2*kappa*(U32(1)^(-(sigma^2)/(a*delta*kappa*(kappa-delta)))-1);
            %%
            % Paso 2. es la creación de U2, compensada con U32(2).
            %%
            % Paso 3.
            TU2 = (0.5*(kappa+delta)/kappa)^((0.5*(kappa+delta)/kappa)*2*a*delta/(sigma^2))*((W_g+1)/(2*kappa/(kappa+delta)+W_g))^(a*delta*(kappa+delta)/(kappa*sigma^2))*(W_g/(W_g+1));

            compara = (U32(2)<=TU2);
            %c = c+1;
            if(compara==true)
               S = (log(W_g+1))/kappa;
               break
            end 
        end
        %%
        % Teorema 3.3
        U3 = rand(1);
        d_im1 = (1+(log(U3)/(2*lambda(i-1)))*(kappa+delta))/(1-(log(U3)/(2*lambda(i-1)))*(kappa-delta));

        if d_im1>0
            V_im1 = -log(d_im1)/kappa;
            S_nm1 = min(S,V_im1);
        elseif d_im1<0
            S_nm1 = S;
        end
        %%
        % simulación $E_{n+1}$
        E_nm1 = -1/rho*log(rand(1));
        %%
        S = min(S,E_nm1);
        T(i) = T(i-1)+S;
        %%
        % Teorema 3.5
        U35 = rand(1,2);

        lambda_poisson = lambda(i-1)*((Es(S)/Bs(S))-(Fs(S)/Cs(S)));
        Js = poissinv(U35(1),lambda_poisson);

        w1s = (D*Bs(S))/(D*Bs(S)+lambda(i-1)*(Es(S)-Fs(S)*(Bs(S)/Cs(S))));

        a1 = Js+D+1;
        a2 = Js+D+2;
        b = Cs(S)/Bs(S);

        if U35(2) < w1s
            lambda_menos = gaminv(U35(2),a1,1/b);
        else
            lambda_menos = gaminv(U35(2),a2,1/b);
        end

        %%
        % Simulación $Y_{i+1}$
        UY = rand(1);
        if min(S_nm1,E_nm1) == S
            Y_im1 = expinv(UY,alpha);
            lambda(i) = lambda_menos + Y_im1;
            saltos(1:3,i) = [T(i);Y_im1;0];
            % Modificar $N$
            N(i) = N(i-1)+1;
        else
           X_im1 = expinv(UY,beta);
           lambda(i) = lambda_menos + X_im1;  
           saltos(1:3,i) = [T(i);0;X_im1];
        end
        %%
    end
    saltos = saltos(:,1:(end-1));
    N = N(1:(end-1));
    lambda = lambda(1:(end-1));
    T = T(1:(end-1));
    %% Construcción de la gráfica completa
    %
    %%
    % Construcción de las observaciones de Brownianos
    %
    % Adecuación de los saltos hasta el periodo de madurez T.
    if max(saltos(1,:))< maturity*anual
        saltos = [saltos,[maturity*anual;0;0]];
    end
    % Constantes
    cantidad = 500;
    saltos_y = saltos(1:2,1:(end-1));
    saltos_x = saltos([1,3],1:(end-1));
    %%
    browniano = [];
    Ws = [];
    cw = 1;
    browniano(:,cw) = [0;0];
    Ws(cw) = 0;
    simus = size(saltos);
    simus = simus(:,2)-1;
    for i = 1:simus
        inf = saltos(1,i);
        sup = saltos(1,i+1);
        t = linspace(inf,sup,cantidad);
        for j = 2:cantidad
            cw = cw+1;
            browniano(:,cw) = [0;0];
            Ws(cw) = norminv(rand(1),0,sqrt(t(j)-t(j-1)));
            browniano(:,cw) = [t(j);Ws(cw)];
        end
    end
    %%
    % Gráfica de $\lambda_{t}$ para los tiempos intermedios.
    k = 0;
    dt = zeros(1,(simus)*cantidad);
    lt = zeros(1,(simus)*cantidad);
    aux_lt = 2*maturity*anual+zeros(2,(cantidad-1)*(length(saltos)-1)+1);
    cl = 1;
    aux_lt(:,cl) = [0;lambda_0];
    dWs = browniano(2,2:end)-browniano(2,1:end-1);
    dWs = [browniano(1,:);dWs,0];
    ts = 1;
    %%
    % Construcción del gráfico
    for i = 1:simus
        inf = saltos(1,i);
        sup = saltos(1,i+1);
        t = linspace(inf,sup,cantidad);
        salto_y = saltos_y(2,1:i);
        salto_x = saltos_x(2,1:i);
        Tiempos = saltos(1,1:i);
        f = @(x) a+(lambda_0-a)*exp(-delta*x)+sum(salto_y.*exp(-delta*(x-Tiempos)))+sum(salto_x.*exp(-delta*(x-Tiempos)));
        for j = 1:length(t)
            if j+k ~= 1
                tsigma = sum(exp(-delta*(t(j)-aux_lt(1,aux_lt(1,:)<t(j)))).*sqrt(aux_lt(2,aux_lt(1,:)<t(j))).*browniano(2,browniano(1,:)<t(j)));
                dt(j+k) = t(j);
                lt(j+k) = f(t(j))+sigma*tsigma;
                if ~any(aux_lt(1,:)==t(j))
                    cl = cl+1;
                    aux_lt(:,cl)=[dt(j+k);lt(j+k)];
                else
                    aux_lt(:,cl)=[dt(j+k);lt(j+k)];
                end
            else
                lt(j+k) = lambda_0;
            end
        end
        k = k+cantidad;
    end
    %%
    % Interpolación para recuperar de la trayectoria los valores en los
    % tiempos observados
    t_obs = 1:(sum((H(:,2)>=FS).*(H(:,2)<=FE))-1);
    N_obs = num_fechas(num_fechas == FS,end);
    t_obs = t_obs./N_obs;
    trayec = zeros(size(t_obs));
    ZZ = size(t_obs);
    ZZ = ZZ(2);
    Z = 0;
    for i = 1:ZZ
%         disp(i)
        if any(floor(dt*10000)==floor(t_obs(i)*10000))
           Z = dt(floor(dt*10000)==floor(t_obs(i)*10000));
        elseif any(floor(dt*10000)==floor(t_obs(i)*10000-1))
           Z = dt(floor(dt*10000)==floor(t_obs(i)*10000-1));
        elseif any(floor(dt*10000)==floor(t_obs(i)*10000-2))
            Z = dt(floor(dt*10000)==floor(t_obs(i)*10000-2));
        elseif any(floor(dt*10000)==floor(t_obs(i)*10000-3))
            Z = dt(floor(dt*10000)==floor(t_obs(i)*10000-3));
        elseif any(floor(dt*10000)==floor(t_obs(i)*10000-4))
            Z = dt(floor(dt*10000)==floor(t_obs(i)*10000-4));
        elseif any(floor(dt*10000)==floor(t_obs(i)*10000-5))
            Z = dt(floor(dt*10000)==floor(t_obs(i)*10000-5));
        end
       
       if any(Z > t_obs(i))
           Z = Z(Z > t_obs(i));
           trayec(i) = min(lt(dt==min(Z(1))));
       else
          Z =  Z(Z < t_obs(i));
          if any(dt==Z(end))
              KK = dt==Z(end);
              KK = dt(KK);
              Z = KK(end);
              trayec(i) = min(lt(dt==min(Z(end))));
          else
              KK = dt==Z(1);
              KK = dt(KK);
              Z = KK(1);
              trayec(i) = min(lt(dt==min(Z(1))));
          end
       end
    end
end