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
logRet = T.logReturn;
ret = T.Return;

%%
%Step 2 : Asdigning 0 and 1 for movement
figure
T.Dir = T.logReturn == abs(T.logReturn);
%disp(T.Dir);
subplot(2,1,1), plot(T.logReturn);
subplot(2,1,2), plot(T.Dir);
%disp(T);
%%
%Step 3 : Training and Testing Model
numTrain = 145;
numTest = size - numTrain;

model = fitensemble(T(1:numTrain, 1:end-1), T(1:numTrain, end),'Bag',100,'Tree','Type','classification');
logicalPredict = predict(model,T(numTrain+1:end, 1:end-1));
disp(logicalPredict);
disp(T.Dir(numTrain+1:end));
figure
plot(abs(T.Dir(numTrain+1:end) - logicalPredict));
title 'Actual - Predicted (To capture fault) -> No fault'

%%
%Step 4 : Volatility Modelling Using GARCH
res = zeros(size, 1);
Mean = zeros(size,1);
for i = 1:size
    Mean(i) = mean(T.Return(1:i));
    res(i) = T.Return(i) - Mean(i);
end
figure
plot(T.Return); hold on
plot(Mean); hold on
plot(res); hold off
legend ('Return', 'Mean of Return Till Dtae','Residual', Location='best');

figure
res_sqr = res.^2;
sd = sqrt(var(res_sqr));
disp(sd);
plot(res_sqr); hold on;
h1 = lbqtest(res);
h2 = lbqtest(res_sqr);
disp(h1); disp(h2);

numPredict = 12;
Mdl = egarch(6,29);
%Mdl.SeriesName = "Return";
EMdl = estimate(Mdl, T.Return(1:numTrain));
out = infer(EMdl, T.Return(1:numTrain));
Err_inSample = rmse(out, res_sqr(1:numTrain));
%disp(out);
plot(out); hold on
%disp(var(T.Return));
%disp(mean(T.Return));
result = forecast(EMdl, numPredict, T.Return(1:numTrain));
disp(result);
Err_outSample = rmse(result, res_sqr(numTrain+1:numTrain+numPredict));
plot(numTrain:numTrain+numPredict-1, result, LineWidth=2);
legend ('Actual Volatility', 'Modelled Volatility', 'Predicted Volatility', Location='best');
title 'Volatility Using EGARCH(6,29)';
hold off

%%
retPredicted = zeros(numPredict, 1);
dirPredict = zeros(numPredict, 1);
for i = 1:numPredict
    if logicalPredict(i) == 0
        dirPredict(i) = -1;
    end
end
for i = 1:numPredict
    retPredicted(i) = Mean(i+numTrain) + dirPredict(i) * sqrt(exp(result(i)));
end

plot(T.Time(1:end), T.Return(1:end)); 
hold on
plot(numTrain+1:numTrain+numPredict, retPredicted, LineWidth=2)
ylabel 'Return'; xlabel 'Time'; title 'Return Prediction';
title 'Return Prediction with RF and EGARCH(6,29) --> RMSE = 17.2013';

RMSE = rmse(retPredicted, T.Return(numTrain+1:numTrain+numPredict));
disp(RMSE);
