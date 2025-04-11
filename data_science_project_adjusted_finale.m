clear 
close all
clc 


% DATA SCIENCE PROJECT
% Time series: NASDAQ-100 index and Apple stock, from 03-01-2012 to 30-12-2022

%data loading:

% Load Data using readmatrix
AAPL = readmatrix('AAPL.xlsx', 'Range', 'E7837:E10604');
NDX = readmatrix('NDX100.xlsx', 'Range', 'E1765:E4532');


% First step: ensure the data lengths match
if length(AAPL) ~= length(NDX)
    error('The lengths of AAPL and NDX data do not match.');
end

% Create timetables for each data series
startDate = datetime(2012, 1, 3);
endate=datetime(2022,12,30)
numDays = length(AAPL);
dates = busdays(startDate,endate)
aapl_data = timetable(dates, AAPL, 'VariableNames', {'Close_AAPL'});
ndx_data = timetable(dates, NDX, 'VariableNames', {'Close_NDX'});
figure
plot(dates,AAPL,Color="r")
title("AAPL")
figure
plot(dates,NDX)
title("NDX")

% Define periods for N1 (stock) and N2 (index)
N_values = [5, 10, 20, 90, 270]; % N1 and N2
M_values = [1, 5, 10, 20, 90, 270]; % M is the lenght of the forecast (in days)
%% the functions:

% volatility
calculate_volatility = @(prices, n) [NaN(n,1); movmean((prices(n+1:end) - prices(1:end-n)) ./ prices(1:end-n), n)];

% momentum
calculate_momentum = @(prices) [NaN(1,1); sign(prices(2:end) - prices(1:end-1))];

% To ensure momentum is 1 or -1
transform_momentum = @(momentum) arrayfun(@(x) 1*(x > 0) + (-1)*(x < 0), momentum);
%% 

% Initialization
results = [];

% Calculate features for each combination of N1 and N2
for N1 = N_values
    for N2 = N_values
        % features for AAPL
        stock_volatility = calculate_volatility(aapl_data.Close_AAPL, N1); %for the volatility
        stock_momentum = transform_momentum(calculate_momentum(aapl_data.Close_AAPL)); %for the momentum
        stock_features = timetable(aapl_data.dates, stock_volatility, stock_momentum, 'VariableNames', {['Stock_Volatility_', num2str(N1)], ['Stock_Momentum_', num2str(N1)]});  
        aapl_data_with_features = synchronize(aapl_data, stock_features); %this function align different tables basing on the row times
        
        % features for NDX
        index_volatility = calculate_volatility(ndx_data.Close_NDX, N2);
        index_momentum = transform_momentum(calculate_momentum(ndx_data.Close_NDX));
        index_features = timetable(ndx_data.dates, index_volatility, index_momentum, 'VariableNames', {['Index_Volatility_', num2str(N2)], ['Index_Momentum_', num2str(N2)]});
        ndx_data_with_features = synchronize(ndx_data, index_features);
        
        % to combine the timetables
        combined_data = synchronize(aapl_data_with_features, ndx_data_with_features);
        
        % To remove NaN values
        combined_data = rmmissing(combined_data); 
        
        % To prepare feature and target vectors:
        %strcat is a built in function to concatenate the strings
        features = [strcat('Stock_Volatility_', string(N1)), strcat('Stock_Momentum_', string(N1)), ...
                    strcat('Index_Volatility_', string(N2)), strcat('Index_Momentum_', string(N2))];
        X = combined_data{:, features};
        y = sign(combined_data.Close_AAPL(2:end) - combined_data.Close_AAPL(1:end-1));
        
        % Ensure the target vectors are correctly aligned
        y = y(1:end);
        X = X(1:end-1, :);
        
        % Remove rows where y is not -1 or 1
        valid_idx = y == -1 | y == 1;
        X = X(valid_idx, :);
        y = y(valid_idx);
        
        % Loop over M values
        for M = M_values
            % we need to shift y by M to predict M days ahead
            y_shifted = circshift(y, -M);
            valid_shifted_idx = y_shifted == -1 | y_shifted == 1;
            y_shifted = y_shifted(valid_shifted_idx);
            X_shifted = X(valid_shifted_idx, :);
            
            % Splitting  data into training and testing sets
            train_ratio = 0.5;
            train_size = round(height(X_shifted) * train_ratio);
            X_train = X_shifted(1:train_size, :);
            y_train = y_shifted(1:train_size);
            X_test = X_shifted(train_size+1:end, :);
            y_test = y_shifted(train_size+1:end);
            
            % Training SVM with Polynomial Kernel
            SVMModel_poly = fitcsvm(X_train, y_train, 'KernelFunction', 'polynomial', 'PolynomialOrder', 3); % to apply svm
            % Training SVM with Gaussian (RBF) Kernel
            SVMModel_rbf = fitcsvm(X_train, y_train, 'KernelFunction', 'rbf', 'KernelScale', 'auto');
            
            
            % Evaluation of Polynomial Kernel SVM
            y_pred_poly = predict(SVMModel_poly, X_test);
            hit_ratio_poly = sum(y_pred_poly == y_test) / length(y_test);
            
            % Evaluation of Gaussian Kernel SVM
            y_pred_rbf = predict(SVMModel_rbf, X_test);
            hit_ratio_rbf = sum(y_pred_rbf == y_test) / length(y_test);
            results = [results; {M, N1, N2, hit_ratio_poly, hit_ratio_rbf}];
            
        end
    end
end



%% Now we can simply convert results to table
results_table = cell2table(results, 'VariableNames', {'M', 'N1', 'N2', 'Hit_Ratio_Polynomial', 'Hit_Ratio_Gaussian'});
disp(results_table);
writetable(results_table, 'SVM_Hit_Ratios.xlsx'); % to store results in a excel file
%% the range

maxhitpol=max(results_table(:,4))
minhitpol=min(results_table(:,4))
maxhitgss=max(results_table(:,5))
minhitgss=min(results_table(:,5))
%% 2d plot  of M and accuracy

unique_M = unique([results_table.M(:)]); % unique is built in function to ensure no repetion of values
hit_ratios_poly = zeros(length(unique_M), 1);
hit_ratios_gss = zeros(length(unique_M), 1);

for i = 1:length(unique_M)
    M_val = unique_M(i);
    rows_with_M = [results_table.M(:)] == M_val;% to filter rows where M equals the current M_val
    % to calculate mean hit ratio
    hit_ratios_poly(i) = mean([results_table.Hit_Ratio_Polynomial(rows_with_M)]);
    hit_ratios_gss(i) = mean([results_table.Hit_Ratio_Gaussian(rows_with_M)]);
end

figure;
plot(unique_M, hit_ratios_poly, '-o', 'DisplayName', 'Polynomial Kernel');
hold on;
plot(unique_M, hit_ratios_gss, '-x', 'DisplayName', 'Gaussian Kernel');
xlabel('Forecast Length (M) in Days');
ylabel('Hit Ratio');
title('Hit Ratio as a Function of Forecast Length (M)');
legend show;
grid on;
hold off;



%% 2d plot  of N1 and accuracy (same process for M)
unique_N1 = unique([results_table.N1(:)]);
hit_ratios_poly_N1 = zeros(length(unique_N1), 1);
hit_ratios_gss_N1 = zeros(length(unique_N1), 1);

for i = 1:length(unique_N1)
    N1_val = unique_N1(i);
    rows_with_N1 = [results_table.N1(:)] == N1_val;
    hit_ratios_poly_N1(i) = mean([results_table.Hit_Ratio_Polynomial(rows_with_N1)]);
    hit_ratios_gss_N1(i) = mean([results_table.Hit_Ratio_Gaussian(rows_with_N1)]);
end

figure;
plot(unique_N1, hit_ratios_poly_N1, '-o', 'DisplayName', 'Polynomial Kernel');
hold on;
plot(unique_N1, hit_ratios_gss_N1, '-x', 'DisplayName', 'Gaussian Kernel');
xlabel('Period Length (N1) in Days');
ylabel('Hit Ratio');
title('Hit Ratio as a Function of Period Length (N1)');
legend show;
grid on;
hold off;


%% 2d plot  of N2 and accuracy (same process for M)
unique_N2 = unique([results_table.N2(:)]);
hit_ratios_poly_N2 = zeros(length(unique_N2), 1);
hit_ratios_gss_N2 = zeros(length(unique_N2), 1);

for i = 1:length(unique_N2)
    N2_val = unique_N2(i);
    rows_with_N2 = [results_table.N2(:)] == N2_val;
    hit_ratios_poly_N2(i) = mean([results_table.Hit_Ratio_Polynomial(rows_with_N2)]);
    hit_ratios_gss_N2(i) = mean([results_table.Hit_Ratio_Gaussian(rows_with_N2)]);
end

figure;
plot(unique_N2, hit_ratios_poly_N2, '-o', 'DisplayName', 'Polynomial Kernel');
hold on;
plot(unique_N2, hit_ratios_gss_N2, '-x', 'DisplayName', 'Gaussian Kernel');
xlabel('Period Length (N2) in Days');
ylabel('Hit Ratio');
title('Hit Ratio as a Function of Period Length (N2)');
legend show;
grid on;
hold off;


%% 3d plot with M and N1 and accuracy


% Initialization of matrices to store hit ratios for 3D plotting
hit_ratios_poly = NaN(length(unique_M), length(unique_N1));
hit_ratios_gss = NaN(length(unique_M), length(unique_N1));


for i = 1:height(results_table)
    M_val = results_table.M(i);
    N1_val = results_table.N1(i);
    M_idx = find(unique_M == M_val);
    N1_idx = find(unique_N1 == N1_val);
    hit_ratios_poly(M_idx, N1_idx) = results_table.Hit_Ratio_Polynomial(i);
    hit_ratios_gss(M_idx, N1_idx) = results_table.Hit_Ratio_Gaussian(i);
end
[M_mesh, N1_mesh] = meshgrid(unique_M, unique_N1); % Create meshgrid for plotting

figure;
surf(M_mesh, N1_mesh, hit_ratios_poly');
xlabel('Forecast Length (M) in Days');
ylabel('Period Length (N1) in Days');
zlabel('Hit Ratio');
title('Hit Ratio for Polynomial Kernel');
colorbar;
grid on;


figure;
surf(M_mesh, N1_mesh, hit_ratios_gss');
xlabel('Forecast Length (M) in Days');
ylabel('Period Length (N1) in Days');
zlabel('Hit Ratio');
title('Hit Ratio for Gaussian Kernel');
colorbar;
grid on;

%% 3d plot with M and N2 and accuracy


% Initialization 
hit_ratios_poly_N2 = NaN(length(unique_M), length(unique_N2));
hit_ratios_gss_N2 = NaN(length(unique_M), length(unique_N2));


for i = 1:height(results_table)
    M_val = results_table.M(i);
    N2_val = results_table.N2(i);
    M_idx = find(unique_M == M_val);
    N2_idx = find(unique_N2 == N2_val);
    hit_ratios_poly_N2(M_idx, N2_idx) = results_table.Hit_Ratio_Polynomial(i);
    hit_ratios_gss_N2(M_idx, N2_idx) = results_table.Hit_Ratio_Gaussian(i);
end

[M_mesh, N2_mesh] = meshgrid(unique_M, unique_N2);
figure;
surf(M_mesh, N2_mesh, hit_ratios_poly_N2');
xlabel('Forecast Length (M) in Days');
ylabel('Period Length (N2) in Days');
zlabel('Hit Ratio');
title('Hit Ratio for Polynomial Kernel');
colorbar;
grid on;

figure;
surf(M_mesh, N2_mesh, hit_ratios_gss_N2');
xlabel('Forecast Length (M) in Days');
ylabel('Period Length (N2) in Days');
zlabel('Hit Ratio');
title('Hit Ratio for Gaussian Kernel');
colorbar;
grid on;
