close all;
clear all;

%data setup
load 'TRAIN02.DAT'
x = [TRAIN02(3001:4998,1) TRAIN02(3001:4998,5)]';
t = [TRAIN02(3003:5000,1) TRAIN02(3003:5000,5)]';
%X = num2cell(x);
%T = num2cell(t);
X = con2seq(x);
T = con2seq(t);

%in = con2seq(X);
%tg = con2seq(T);
%[X,T] = simpleseries_dataset;
%[Xs,Xi,Ai,Ts] = preparets(net,X,T);

%net setup
net = distdelaynet({1:2,1:2},10);
[Xs,Xi,Ai,Ts] = preparets(net,X,T);
% net.trainFcn = 'trainlm';
% net.divideFcn = '';
% net.trainParam.epochs = 1000;
%net.trainParam.goal = 0.0001;

%net = train(net,X,T);
net = train(net,Xs,Ts,Xi,Ai);
%view(net)
P = net(Xs,Xi,Ai);
perf = perform(net,P,Ts);
%plotresponse(T,P);

I = cell2mat(X);
O = cell2mat(P);
figure;
plot(I(1,:)');
hold on
plot(O(1,:)','--');
hold off
