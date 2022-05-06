function [dt,lt] = f_trayectoria_55(a,lambda_0,delta,sigma,beta,alpha,rho,maturity,anual,nombre)
% lambda_0 =6.84;
% a =  xx(1);	delta = xx(2);
% sigma = xx(3);    beta = xx(4);   alpha = xx(5);  rho = xx(6);
% maturity = 1;
% anual = 1;
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
%     hold on
    plot(dt,lt)
    % plot(dt(dt<=0.1),lt(dt<=0.1))
    %%
    %% Importación de los datos del gráfico de $\lambda_{t}$
    gt = dt';
    gl = lt';
    G = table(gt,gl);
    writetable(G,nombre);
end


