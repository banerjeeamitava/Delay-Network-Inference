clear
rng('default')
rng shuffle
warning('off','all')
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 1: Simulation of the delayed 4-laser network dynamics %%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Library of all possible 4-node connected directed networks (62 networks in total)
%Following https://arxiv.org/abs/1106.3994v1 page 12, with following node numbering for diamond-shaped 4-node network
%1-top, 2-middle left, 3-middle right, 4-bottom

%Later in the code, you'll have an option to choose your adjacency matrix
%from this list

A=zeros(62,4,4);% A will store adjacency matrices of 62 4-node networks
%m=number of links below
%% m=3

A(1,:,:)=[0,0,0,0; 1,0,0,0; 1,0,0,0; 1,0,0,0;];

A(2,:,:)=[0,0,0,0; 1,0,0,0; 0,1,0,0; 0,1,0,0;];
A(3,:,:)=[0,0,0,0; 1,0,0,0; 1,0,0,0; 0,0,1,0;];

A(4,:,:)=[0,0,0,0; 1,0,0,0; 0,1,0,0; 0,0,1,0;];

%% m=4

A(5,:,:)=[0,0,0,1; 1,0,0,0; 1,0,0,0; 1,0,0,0;];
A(6,:,:)=[0,0,0,0; 1,0,0,0; 1,0,0,0; 1,0,1,0;];
A(7,:,:)=[0,0,1,0; 1,0,0,0; 1,0,0,0; 0,0,1,0;];
A(8,:,:)=[0,0,0,1; 1,0,1,0; 0,0,0,1; 0,0,0,0;];

A(9,:,:)=[0,0,1,0; 0,0,1,1; 0,0,0,1; 0,0,0,0;];
A(10,:,:)=[0,0,0,1; 0,0,1,0; 0,1,0,0; 0,0,1,0;];
A(11,:,:)=[0,0,1,1; 0,0,0,0; 0,1,0,0; 0,0,1,0;];
A(12,:,:)=[0,0,1,0; 0,0,0,1; 0,1,0,1; 0,0,0,0;];
A(13,:,:)=[0,0,1,1; 0,0,0,1; 0,1,0,0; 0,0,0,0;];

%% m=5

A(14,:,:)=[0,0,0,1; 1,0,0,1; 1,0,0,0; 1,0,0,0;];
A(15,:,:)=[0,0,0,0; 1,0,1,0; 1,0,0,0; 1,0,1,0;];
A(16,:,:)=[0,0,0,1; 1,0,1,0; 1,0,0,0; 1,0,0,0;];
A(17,:,:)=[0,1,1,0; 0,0,1,0; 0,1,0,0; 1,0,0,0;];

A(18,:,:)=[0,0,0,1; 1,0,0,0; 1,0,0,0; 1,0,1,0;];
A(19,:,:)=[0,1,0,0; 0,0,1,0; 1,1,0,0; 1,0,0,0;];
A(20,:,:)=[0,0,1,1; 1,0,0,0; 1,0,0,0; 0,0,1,0;];
A(21,:,:)=[0,0,0,0; 1,0,1,0; 1,0,0,0; 1,1,0,0;];
A(22,:,:)=[0,0,0,1; 1,0,1,0; 1,0,0,1; 0,0,0,0;];
A(23,:,:)=[0,0,1,1; 0,0,1,0; 0,1,0,0; 0,1,0,0;];
A(24,:,:)=[0,1,1,0; 0,0,0,1; 1,0,0,0; 1,0,0,0;];

%% m=6

A(25,:,:)=[0,0,0,1; 1,0,0,1; 1,0,0,1; 1,0,0,0;];

A(26,:,:)=[0,0,0,1; 1,0,1,0; 1,0,0,0; 1,0,1,0;];
A(27,:,:)=[0,1,0,0; 1,0,1,0; 1,0,0,0; 1,1,0,0;];
A(28,:,:)=[0,0,0,1; 1,0,1,0; 1,0,0,1; 1,0,0,0;];
A(29,:,:)=[0,1,1,0; 1,0,0,1; 0,1,0,0; 1,0,0,0;];

A(30,:,:)=[0,0,1,1; 0,0,1,0; 0,1,0,1; 0,1,0,0;];
A(31,:,:)=[0,0,1,1; 0,0,1,0; 1,0,0,0; 0,1,1,0;];
A(32,:,:)=[0,1,0,0; 0,0,1,1; 1,0,0,1; 0,0,1,0;];
A(33,:,:)=[0,1,0,0; 0,0,1,0; 1,0,0,1; 0,1,1,0;];

%% m=7

A(34,:,:)=[0,0,1,1; 1,0,0,1; 1,0,0,1; 1,0,0,0;];
A(35,:,:)=[0,0,0,1; 1,0,1,0; 1,0,0,1; 1,0,1,0;];
A(36,:,:)=[0,1,0,0; 1,0,0,0; 1,1,0,0; 1,1,1,0;];
A(37,:,:)=[0,1,1,0; 1,0,0,1; 1,0,0,1; 1,0,0,0;];

A(38,:,:)=[0,0,1,1; 1,0,1,0; 1,0,0,1; 1,0,0,0;];
A(39,:,:)=[0,0,0,1; 1,0,1,1; 1,0,0,0; 1,0,1,0;];
A(40,:,:)=[0,1,1,0; 1,0,1,0; 1,0,0,1; 1,0,0,0;];
A(41,:,:)=[0,1,0,0; 1,0,1,0; 1,0,0,1; 1,0,1,0;];
A(42,:,:)=[0,1,1,0; 1,0,0,1; 1,0,0,0; 1,0,1,0;];
A(43,:,:)=[0,1,0,0; 1,0,1,1; 1,0,0,0; 1,0,1,0;];
A(44,:,:)=[0,1,1,0; 1,0,0,1; 1,1,0,0; 0,0,1,0;];

%% m=8

A(45,:,:)=[0,0,1,1; 1,0,1,1; 1,0,0,1; 1,0,0,0;];
A(46,:,:)=[0,1,1,0; 1,0,1,0; 1,1,0,0; 1,1,0,0;];
A(47,:,:)=[0,1,1,0; 1,0,0,1; 1,1,0,0; 1,1,0,0;];
A(48,:,:)=[0,1,0,0; 1,0,1,1; 1,0,0,1; 1,0,1,0;];

A(49,:,:)=[0,1,1,0; 1,0,1,1; 1,0,0,0; 1,0,1,0;];
A(50,:,:)=[0,1,1,0; 1,0,0,1; 1,0,0,1; 1,0,1,0;];
A(51,:,:)=[0,1,1,0; 1,0,1,1; 1,0,0,1; 1,0,0,0;];
A(52,:,:)=[0,0,1,0; 1,0,0,1; 1,1,0,0; 1,1,1,0;];
A(53,:,:)=[0,1,0,0; 1,0,1,1; 1,1,0,0; 1,0,1,0;];

%% m=9

A(54,:,:)=[0,0,1,1; 1,0,1,1; 1,0,0,1; 1,0,1,0;];

A(55,:,:)=[0,1,1,0; 1,0,0,1; 1,1,0,0; 1,1,1,0;];
A(56,:,:)=[0,1,1,0; 1,0,1,1; 1,1,0,0; 1,0,1,0;];

A(57,:,:)=[0,1,0,1; 1,0,1,1; 1,1,0,0; 1,0,1,0;];

%% m=10

A(58,:,:)=[0,1,1,1; 1,0,1,1; 1,0,0,1; 1,0,1,0;];
A(59,:,:)=[0,1,1,1; 0,0,1,1; 1,1,0,0; 1,1,1,0;];

A(60,:,:)=[0,1,1,1; 1,0,1,1; 1,1,0,0; 1,0,1,0;];

%% m=11

A(61,:,:)=[0,1,1,1; 1,0,1,1; 1,0,0,1; 1,1,1,0;];

%% m=12

A(62,:,:)=[0,1,1,1; 1,0,1,1; 1,1,0,1; 1,1,1,0;];


Adata=A;

%% Parameters for the simulation of 4-node laser networks

% Filter Parameters
fl=100;     % HPF cut-on
fh=2500;    % LPF cut-off
Fs=24e3;    % Sampling rate
[b,a]=butter(1,[fl fh]*2/Fs); %get filter coefficients
a1=a(2);
a2=a(3);
b0=b(1);
b2=b(3);

% More Simulation parameters
N=5e5; % Total simulation number of steps

Nnoise=100; % Step that feedback is turned on
Nfb=9600; % Number of steps that feedback is on before coupling is enabled
Nswitch=Nnoise+Nfb; % Step that coupling is turned on

%these are parameters that can lead to chimera states in globally coupled 4
%node networks. As we discussed on Zoom, it may be preferably to set kc=kf
%for this work.
beta = -3.8; %feedback strength; MUST be negative
epsilon = 1.5; %coupling strength (sigma in my thesis)
kf = 34; %feedback delay in time steps, minimum is 34
kc = 34; %coupling delay in time steps
nnodes=4; % number of nodes
phi=pi/4; %phi can be varied, but we typically leave it at pi/4

%Build the Adjacency Matrix
B = ones(nnodes) - eye(nnodes); %this is all-to-all coupling. We have the capability to do any coupling topology in our experiments.
C(:,:)=Adata(10,:,:);%Coupling Topology (here we use a particular network from the list of 62 networks above)
A=B.*C;
ModelParams.Ad=A;
NoiseD=10^-3;%strength of the dynamical noise, applied independently at each node and at each time step

%% Initiate system in a random state

%x will be the variable that save the time-series from the 4 nodes
x(1:nnodes,1:Nnoise)=0.0256+ 0.0394.*randn(nnodes,Nnoise);  % Noise for the first Nnoise steps

%% Let the individual lasers evolve with self-feedback w/o any coupling

% Feedback ON, Coupling OFF i.e. epsilon=0
for ll=Nnoise+1:Nswitch
    for kk=1:nnodes
        r = cos ( x(kk,ll-kf)+phi )^2 ;
        r2 = cos ( x(kk,ll-kf-2)+phi )^2 ;
        x(kk,ll)=-a1*x(kk,ll-1)-a2*x(kk,ll-2) + beta*b0*(r-r2);
    end
end
y=x;

%% Final stages of simulation: Feedback ON, Coupling ON 

% The following variables will store time-series data and derivative data
diff_data=zeros(nnodes,N-Nswitch);
data=zeros(nnodes,N-Nswitch);

%Refer to PHYSICAL REVIEW X 11, 031014 (2021) Appendix for the equations
%(there are two equvalent versions of the model equations: (1) a continuous-time one (Eq. 10) that is conceptually
%easy to understand, and used in the reservoir computer formalism, and (2)
%a discrete-time one that is useful for fast simulation of the system and used in the following lines of this code,
% it is described in Appendix B)
for ll=(Nswitch+1):N %time loop
    
for kk=1:nnodes
        x(kk,ll) = y(kk,ll-kf) + epsilon*( A(kk,:)*( y(:,ll-kf)- y(kk,ll-kf)) ) + sqrt(2*NoiseD)*randn(1,1);     
        r = cos ( x(kk,ll) + phi )^2 ;
        r2 = cos ( x(kk,ll-2) + phi )^2 ;
        y(kk,ll)=-a1*y(kk,ll-1)-a2*y(kk,ll-2)+beta*b0*(r-r2);
end   


diff_data(1:nnodes,ll-Nswitch)=x(:,ll);%stored Data for derivative (reservoir output)
data(1:nnodes,ll-Nswitch)=x(:,ll-kc);%Apply the time-delay to data, to use it as reservoir input

end

%% Discard transients so that you only collect data from the attractor
data0=data(:,10000:end);
diff_data0=diff_data(:,10000:end);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    Part 2: Using Reservoir Computer for Network Inference  %%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input and outputs of the reservoir
ModelParams.N=nnodes;

% choose a segment of data
data=data0(:,1:5e4);
diff_data=diff_data0(:,1:5e4);


measurements1 = data;% + z;
measurements2 = diff_data;% + z;

%% Specifications of the Reservoir Computer
 resparams=struct();
%train reservoir
[num_inputs,~] = size(measurements1);
resparams.radius = 0.9; % spectral radius
resparams.degree = 2.38; % connection degree
approx_res_size = 3000; % reservoir size
resparams.N = floor(approx_res_size/num_inputs)*num_inputs; % actual reservoir size divisible by number of inputs
resparams.sigma = 1.17; % input weight scaling
resparams.train_length = 30000; % number of points used to train
resparams.num_inputs = num_inputs; 
resparams.predict_length = 2000; % number of predictions after training
resparams.predict_length_max=resparams.predict_length;
resparams.beta = 0.0001; %regularization parameter

%% Training of the reservoir
[xx, w_out, A, win,r] = train_reservoir(resparams, measurements1,measurements2);%Train and save w_out matrix for future use

%% Calculation and time-averaging of the Jacobian matrix (Eq. 9 in paper)
av_length=1000;
conn1=zeros(ModelParams.N);
pin=pinv(w_out);
%loop for time-averaging the Jacobian matrix (Eq. 9 in paper)
for it=resparams.train_length-av_length:10:resparams.train_length
    

 Rr=zeros(resparams.N);
 Ru=zeros(resparams.N,ModelParams.N);
 
 for i1=1:resparams.N
     Rr(:,i1)=(sech(r(:,it)).^2).*A(:,i1);
 end
    
 for i1=1:ModelParams.N
     Ru(:,i1)=(sech(r(:,it)).^2).*win(:,i1);
 end
    conn1=conn1+abs((eye(ModelParams.N)-w_out*Rr*pin)\(w_out*Ru))/(av_length/10);
end

%% Store the inferred connectivity matrix diffxy
diffxy=conn1(1:1:nnodes,1:1:nnodes);
diffxy=diffxy-diag(diag(diffxy));

%% %%%%%%%%%%%
%% Plotting %%
%% %%%%%%%%%%%
figure()

subplot(1,2,1)
s1 = (1:1:nnodes);
s2 = (1:1:nnodes);
imagesc(s1,s2,(ModelParams.Ad-diag(diag(ModelParams.Ad))))
colorbar
title('Actual Connections')
pbaspect([1,1,1])
set(gca,'FontSize',25)

subplot(1,2,2)
imagesc(s1,s2,diffxy)
colorbar
title('Inferred Connections')
pbaspect([1,1,1])
set(gca,'FontSize',25)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, wout, A, win,states] = train_reservoir(resparams, data1,data2)

A = generate_reservoir(resparams.N, resparams.radius, resparams.degree);
% A=zeros(resparams.N);%Setting A to zero
q = resparams.N/resparams.num_inputs;
win = zeros(resparams.N, resparams.num_inputs);
for i=1:resparams.num_inputs
%     rng(i,'twister')
    ip = resparams.sigma*(-1 + 2*rand(q,1));
    win((i-1)*q+1:i*q,i) = ip;
end
%win = resparams.sigma*(-1 + 2*rand(resparams.N, resparams.num_inputs));
states = reservoir_layer(A, win, data1, resparams);
wout = train(resparams, states, data2(:,1:resparams.train_length));
x = states(:,end);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = generate_reservoir(size, radius, degree)
%  rng(1,'twister');
sparsity = degree/size;
while 1
A = sprand(size, size, sparsity);
e = max(abs(eigs(A)));

if (isnan(e)==0)%Avoid NaN in the largest eigenvalue, in case convergence issues arise
    break;
end

end
A = (A./e).*radius;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function states = reservoir_layer(A, win, input, resparams)

states = zeros(resparams.N, resparams.train_length);
for i = 1:resparams.train_length-1
    states(:,i+1) = tanh(A*states(:,i) + win*input(:,i));
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w_out = train(params, states, data)

beta = params.beta;
% rng(2,'twister');
idenmat = beta*speye(params.N);
% states(2:2:params.N,:) = states(2:2:params.N,:).^2;
w_out = data*transpose(states)*pinv(states*transpose(states)+idenmat);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [output,r] = predict(A,win,resparams,x, data,w_out)

output = zeros(resparams.num_inputs, resparams.predict_length);
r=zeros(size(x));
for i = 1:resparams.predict_length
    
    p=A*x + win*data(:,i);
    x = tanh(p);
    r=r+sech(p).^2;
    out = w_out*x;
    output(:,i) = out;
end
r=r/resparams.predict_length;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f=residual(x,ModelParams,data1,data2)
measured_vars = 1:1:ModelParams.N;
num_measured = length(measured_vars);
measurements1 = data1(measured_vars, :);% + z;
measurements2 = data2(measured_vars, :);% + z;

resparams=struct();
%train reservoir
[num_inputs,~] = size(measurements1);
resparams.radius = 0.9; % spectral radius, around 0.9
resparams.degree = x(1); % connection degree, around 3
approx_res_size = 3000; % reservoir size
resparams.N = floor(approx_res_size/num_inputs)*num_inputs; % actual reservoir size divisible by number of inputs
resparams.sigma = x(2); % input weight scaling, around 0.1
resparams.train_length = 30000; % number of points used to train
resparams.num_inputs = num_inputs; 
resparams.predict_length = 1000; % number of predictions after training
resparams.beta = 0.0001; %regularization parameter, near 0.0001

[~, H, ~, ~,r] = train_reservoir(resparams, measurements1,measurements2);
output1=H*r;
f=sum((output1(:,floor(resparams.train_length*0.1):resparams.train_length)-data2(:,floor(resparams.train_length*0.1):resparams.train_length)).^2,'all');
end