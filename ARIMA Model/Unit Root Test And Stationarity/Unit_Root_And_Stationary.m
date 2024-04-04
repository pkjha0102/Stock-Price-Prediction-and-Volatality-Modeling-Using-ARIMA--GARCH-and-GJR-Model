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
    T.CloseDiff(2:end) = data1;
    T.CloseDiff = T.Close;

    %Step 2: Checking For Stationarity - Augmented DF Test
    [h1, pvalue1, stat1, cValue1] = adftest(data1);
    disp('new h value is:'); disp(h1);
    % The result h = 1 indicates that this test 
    % rejects the null hypothesis of a unit root 
    % against the autoregressive alternative.


%Step 4: Getting ACF and PACF
    % Using econometricModeler app, PACF lags = 1, ACF lags = 47