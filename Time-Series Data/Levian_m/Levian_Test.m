clear all
close all

load 'TRAIN02.DAT' 

% X1 = normc(TRAIN02(:,1));
% X2 = normc(TRAIN02(:,5));

X1 = TRAIN02(:,1);
X2 = TRAIN02(:,5);

% X1 = normc(X1);
% X2 = normc(X2);

%Given Signals 'X1' and 'X2'

figure(1)
plot(X1);
hold on
plot(X2,'-r');
hold off

%Training with the first 3000 samples of the given data

StepSize = 1;

S1 = X1(1:StepSize:3000);
S2 = X1(2:StepSize:3001);
S3 = X2(1:StepSize:3000);
S4 = X2(2:StepSize:3001);

S5 = [S1 S2 S3 S4]';

T = X1(3:StepSize:3002)';

save S5
save T         %Saving our Inputs and Outputs for Training Phase

time1 = clock;   %Clocking the begin time so as to calculate the time taken

a1 = minmax(S1');
a2 = minmax(S2');
a3 = minmax(S3');
a4 = minmax(S4');

Min_Max = [a1; a2; a3; a4];   %Min-max of the data for scaling purpose in the network

net = newff(Min_Max,[100 1],{'tansig' 'purelin'},'trainlm');    %Using a feedforward network with 100 hidden layers

net.trainParam.goal = 0.00001;
net.trainParam.epochs = 1000;
net.performFcn = 'mae';
%net.plotFcns = {'plotPerform','plottrainstate','ploterrhist','plotfit','plotregression'};


net = train(net,S5,T);

save 'Train02DataNN' 
save net 
save X1 
save X2

%Testing remaining 2000 nodes in the data

StepSize = 1;

S1 = X1(3001:StepSize:4998);
S2 = X1(3002:StepSize:4999);
S3 = X2(3001:StepSize:4998);
S4 = X2(3002:StepSize:4999);

S6 = [S1 S2 S3 S4]';

R = X1(3003:StepSize:5000)';

Ra = sim(net,S6);

save S6

time2 = clock;

figure(2)
plot(R);
hold on
plot(Ra,':r');
hold off

%Calculating Total Time Taken in secs

Total_Time_Secs = (time2(5) - time1(5)) * 60 + (time2(5) - time1(6));

%Calculating the Mean Absolute Prediction Error (MAPE)
%R actual output
%Ra predicted output

Er = Ra - R;
figure(3)
plot(Er)

R = R + 1;         %Doing this step as MAPE doesnt take '0''s as input which leads to infinite error
Ra = R + 1;
MAPE = errperf(R,Ra,'mape');
MSPE = errperf(R,Ra,'mspe');    %Comparison of the Error Percentages

save R 
save Ra 
save MAPE
save MSPE