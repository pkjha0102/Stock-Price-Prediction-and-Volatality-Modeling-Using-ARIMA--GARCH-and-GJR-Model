clc
close all

%Step 1: Importing Data
    T = readtable('ADANIENT.NS.csv');
    T(1:5, 1:5);
    T1 = T.Close;     %Our closing price data
    data = T.Close;   %Log of closing price data
    logData = T.logClose;
    time = T.Time;
    size = height(data);
    T.Return = zeros(size, 1);
    T.Return(2:end, :) = diff(T.Close)./T.Close(1:end-1,:) * 100;
    T.logReturn = zeros(size, 1);
    T.logReturn(2:end) = diff(T.logClose) * 100;
    retData = T.Return;
    figure
    subplot(2,2,1), plot(T.Close); title 'Closing Price'
    subplot(2,2,2), plot(T.logClose); title 'log of Closing Price'
    subplot(2,2,3), plot(T.Return); title 'Return'
    subplot(2,2,4), plot(T.logReturn); title 'log of Return'
    %disp(data(1:10));
    %plot(data); title("SBI daily closing prices fron march 4, 2023 to march 4, 2024

%Step 2: Checking For Stationarity - Augmented DF Test
    [h, pvalue, stat, cValue] = adftest(data);
    disp('h value is:'); disp(h);
    % The result h = 0 indicates that this test 
    % fails to reject the null hypothesis of a unit root 
    % against the autoregressive alternative.

%Step 3: Removing Non-Stationarity
    data1 = diff(T.logClose);
    T.CloseDiff = zeros(size,1);
    T.CloseDiff(2:end) = diff(T.Close);
    T.logCloseDiff = zeros(size,1);
    T.logCloseDiff(2:end) = data1;
    logCloseDiff = T.logCloseDiff;

    %Step 2: Checking For Stationarity - Augmented DF Test
    [h1, pvalue1, stat1, cValue1] = adftest(data1);
    disp('new h value is:'); disp(h1);
    % The result h = 1 indicates that this test 
    % rejects the null hypothesis of a unit root 
    % against the autoregressive alternative.

%Step 4: Getting ACF and PACF
    % Using econometricModeler app, PACF lags = 1, ACF lags = 47

%Step 5: Create Model For Estimation
    Md1 = arima(42,0,0);
    Md1.SeriesName= "Return";

%Step 6: Partiton Data
    numTrain = 143;
    numTest = size - numTrain;

%Estimate Parameters
    EstMd1 = estimate(Md1, T(1:numTrain, :));
    resdt = infer(EstMd1, T(1:numTrain, :));
    tail(resdt);
    resdtt = T.Return(1:numTrain, :) - resdt.Return_Residual;
    tail(resdtt);
    figure
    plot(T.Time(1:numTrain), T.Return(1:numTrain), T.Time(1:numTrain), resdtt); 
    ylabel 'Return'; xlabel 'Time'; title 'ARIMA fitting';
    hold on
    %disp(T);
%%
%Forecast for 15 days
    [dataForecasted] = forecast(EstMd1, numTest+1, T(1:numTrain, :));

%Plot Difference
    actual = T.Return(numTrain:end);
    p1 = plot(T.Time, T.Return); hold on
    result = dataForecasted.Return_Response;
    p2 = plot(T.Time(numTrain:end), result, LineWidth=1.5);
    grid on
    hold on

%Calclate RMSE
    error = rmse(result, actual);
    disp("RMSE is:"); disp(error);

%Plot 95% Confidence Interval
    lower = dataForecasted.Return_Response - 1.96*sqrt(dataForecasted.Return_MSE);
    upper = dataForecasted.Return_Response + 1.96*sqrt(dataForecasted.Return_MSE);
    p3 = plot(T.Time(numTrain:end), [lower, upper], LineStyle="--", LineWidth=2);
    legend([p1, p2, p3(1)], "Observations", "Predicted", "95% confidence interval", Location="best");
    title 'Predicted return at AR(42), RMSE is 21.4739';

    




