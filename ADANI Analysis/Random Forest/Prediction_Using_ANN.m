%clc
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
size = 130;
numTrain = 120;
numTest = size - numTrain;

prediction = zeros(numTest, 1);
window = numTrain; start = 1;
for i = 1:numTest
    model = fitrnet(T(1:window, 2:4), T(1:window, 5));
    prediction(i) = predict(model,T(window+1, 2:4));
    window = window + 1;
    start = start + 1;
end
disp(T.Close(numTrain:size));
disp(prediction);
retPredict = zeros(numTest, 1);
retPredict(1) = ((prediction(1) - T.Close(numTrain)) / T.Close(numTrain)) * 100;
for i = 2:numTest
    retPredict(i) = ((prediction(i) - prediction(i-1)) / prediction(i-1)) * 100;
end
disp(T.Return(numTrain+1:size));
disp(retPredict);
RMSE_Ret = rmse(T.Return(numTrain+1:size), retPredict);
disp(RMSE_Ret);

figure,
plot(T.Return(numTrain+1:size)); hold on
plot(retPredict);
title 'Actual vs Predicted Return (Zoomed) -> RMSE = '
legend('Actual', 'Predicted');

figure,
plot(T.Close(numTrain+1:size)); hold on
plot(prediction);
title 'Actual vs Predicted Closing Price (Zoomed) -> RMSE = '
legend('Actual', 'Predicted');
figure,
plot(T.Close); hold on
plot(numTrain+1:size, prediction);
title 'Actual vs Predicted Closing Price'
legend('Actual', 'Predicted');

RMSE = rmse(prediction, T.Close(numTrain+1:size));
disp(RMSE);