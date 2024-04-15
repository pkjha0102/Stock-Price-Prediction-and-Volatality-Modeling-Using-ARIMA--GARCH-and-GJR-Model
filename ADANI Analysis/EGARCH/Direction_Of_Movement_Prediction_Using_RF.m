clc
close all
warning off

%Step 1: Importing Data
T = readtable('ADANIENT.NS.csv');
data = T.Close;   %Log of closing price data
logData = T.logClose;
size = height(data);
T.Return = zeros(size, 1);
T.Return(2:end, :) = diff(T.Close)./T.Close(1:end-1,:) * 100;
% disp(T.Return);
subplot(2,1,1), plot(T.Return);
title 'Return'
T.logReturn = zeros(size, 1);
T.logReturn(2:end) = diff(T.logClose) * 100;
% disp(T.logReturn);
subplot(2,1,2), plot(T.logReturn);
title 'Log Return'

%%
%Step 2 : Asdigning 0 and 1 for movement
T.Dir = T.logReturn == abs(T.logReturn);
%disp(T.Dir);
plot(T.logReturn); hold on
plot(T.Dir);
%disp(T);
%%
%Step 3 : Training and Testing Model
numTrain = 143;
numTest = size - numTrain;

model = fitensemble(T(1:numTrain, 1:end-1), T(1:numTrain, end),'Bag',100,'Tree','Type','classification');
prediction = predict(model,T(numTrain+1:end, 1:end-1));
disp(prediction);
disp(T.Dir(numTrain+1:end));
plot(abs(T.Dir(numTrain+1:end) - prediction));
title 'Actual - Predicted (To capture fault) -> No fault'

%%
%Step 4 : Volatility Modelling Using GARCH
