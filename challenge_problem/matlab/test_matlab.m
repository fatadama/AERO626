clear variables;
close all;

% epsilon_eqom nonlinear parameter in goerning equation
epsilon_eqom = 1.0e-2;
% a_0 amplitude of forcing costine term
a_0 = 2.0;
% omega_t frequency of forcing function
omega_t = 1.25;
% r_w std of the stochastic forcing term in the nonlinear goerning dynamics
r_w = 1.0;
% tf final time
tf = 10.0;
% x0 initial state
x0 = [0.0 0.0];
% Np number of points
Np = 64;
% ND output time array grid number of points
ND = round(1000*tf);
% scatter in initial x(1)
sigma_x1 = 5;
% scatter in initial x(2)
sigma_x2 = 1;

eqom = @(t,x)[x(2);x(1)-epsilon_eqom*x(1)^3 + randn*r_w];

tspan = linspace(0,tf,ND);

X0 = repmat(x0,Np,1) + randn(Np,2).*repmat([sigma_x1 sigma_x2],Np,1);

XK = zeros(ND,Np*2);
tic
for k = 1:Np
    [T,Y] = ode45(eqom,tspan,X0(k,:));
    XK(:,(1:2) + ((k-1)*2)) = Y;
end
toc
%%
figure();
subplot(221);
plot(tspan,XK(:,1:2:(2*Np-1)));

subplot(222);
plot(tspan,XK(:,2:2:(2*Np)));

subplot(2,2,[3 4]);
plot(XK(:,1:2:(2*Np-1)),XK(:,2:2:(2*Np)));
%%
save data.mat tspan XK;
fprintf('Wrote to file data.mat');