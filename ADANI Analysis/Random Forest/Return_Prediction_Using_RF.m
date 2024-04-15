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

%Step 2 : Training and Testing Model
numTrain = 145;
numTest = size - numTrain;

model = fitensemble(T(1:numTrain, 1:end-1), T(1:numTrain, end),'Bag',100,'Tree','Type','classification');
prediction = predict(model,T(numTrain+1:end, 1:end-1));
disp(prediction);
figure,
plot(T.logReturn(numTrain+1:end)); hold on
plot(prediction);
title 'Actual vs Predicted(Zoomed) -> RMSE = 15.8323'
figure,
plot(T.logReturn); hold on
plot(numTrain+1:size, prediction);
title 'Actual vs Predicted Return'

RMSE = rmse(prediction, T.logReturn(numTrain+1:end));
disp(RMSE);