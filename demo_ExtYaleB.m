%  This is a collection of MATLAB files accompanying the paper
% "Fast Optimization Algorithm on Riemannian Manifolds and Its Application in  Low-Rank Learning"
%     by Haoran Chen, Yanfeng Sun,Junbin Gao and Yongli Hu
%  Here, we appreciate Bamdev Mishra's work "manopt" 
% If there are any problems or bugs, feel free to email me at hr_chen@emails.bjut.edu.cn

clc;
clear;
addpath(genpath('Manopt_1.0.7/.'));
addpath  ncut_toolbox

load('ExtYaleB.mat')
K =2;

X0 = X0(:,1:64*K);
X0 = X0  - repmat(mean(X0,1),size(X0,1),1);


lambda = 0.001;   
rho = 0.5;
begin =tic;
ACz = mytest(X0,label(1:64*K),K,lambda,rho);

errs =1-max(ACz);
endtime = toc(begin)

