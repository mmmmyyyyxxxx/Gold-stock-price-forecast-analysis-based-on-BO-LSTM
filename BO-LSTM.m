clc; clear;
close all;
opt.Delays = 1:30;
opt.dataPreprocessMode  = 'Data Standardization'; 
% "无”“数据标准化”“数据规范化” 
opt.learningMethod      = 'LSTM';
opt.trPercentage        = 524/627;                  
% 将数据分为测试和训练数据集
% ---- 通用深度学习参数(LSTM和CNN通用参数)
opt.maxEpochs     = 400;                         
% 深度学习算法中训练Epoch的最大数目。
opt.miniBatchSize = 32;                         
% 深度学习算法中的最小批处理大小。
opt.executionEnvironment = 'cpu';                
% 'cpu' 'gpu' 'auto'
opt.LR                   = 'adam';               
% 'sgdm' 'rmsprop' 'adam'
opt.trainingProgress     = 'none';  
% 'training-progress' 'none'“训练进步”“没有”
% ------------- BILSTM参数
opt.isUseBiLSTMLayer  = true;                     
% 如果为真，则层转向双向LSTM，如果为假，则将单元转向简单LSTM
opt.isUseDropoutLayer = true;                   
% 退出层避免过度拟合
opt.DropoutValue      = 0.5;
% ------------ 优化参数
opt.optimVars = [
    optimizableVariable('NumOfLayer',[1 4],'Type','integer')
    optimizableVariable('NumOfUnits',[50 200],'Type','integer')
    optimizableVariable('isUseBiLSTMLayer',[1 2],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-2 1],'Transform','log')
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];
opt.isUseOptimizer         = true;
opt.MaxOptimizationTime    = 14*60*60;
opt.MaxItrationNumber      = 60;
opt.isDispOptimizationLog  = true;
opt.isSaveOptimizedValue       = false;        
%  保存所有的优化输出在mat文件
opt.isSaveBestOptimizedValue   = true;         
%  保存最佳优化输出文件
data = loadData(opt);
if ~data.isDataRead
    return;
end
[opt,data] = PrepareData(opt,data);
[opt,data] = OptimizeLSTM(opt,data);
[opt,data] = EvaluationData(opt,data);
function data = loadData(opt)
[chosenfile,chosendirectory] = uigetfile({'*.xlsx';'*.csv'},...
    'Select Excel time series Data sets','data.xlsx');
filePath = [chosendirectory chosenfile];
if filePath ~= 0
    data.DataFileName = chosenfile;
    data.CompleteData = readtable(filePath);
    if size(data.CompleteData,2)>1
        warning('Input data should be an excel file with only one column!');
        disp('Operation Failed... '); pause(.9);
        disp('Reloading data. ');     pause(.9);
        data.x = [];
        data.isDataRead = false;
        return;
    end
    data.seriesdataHeder = data.CompleteData.Properties.VariableNames(1,:);
    data.seriesdata = table2array(data.CompleteData(:,:));
    disp('Input data successfully read.');
    data.isDataRead = true;
    data.seriesdata = PreInput(data.seriesdata);
    
    figure('Name','InputData','NumberTitle','off');
    plot(data.seriesdata); grid minor;
    title({['Mean = ' num2str(mean(data.seriesdata)) ', STD = ' num2str(std(data.seriesdata)) ];});
    if strcmpi(opt.dataPreprocessMode,'None')
        data.x = data.seriesdata;
    elseif strcmpi(opt.dataPreprocessMode,'Data Normalization')
        data.x = DataNormalization(data.seriesdata);
        figure('Name','NormilizedInputData','NumberTitle','off');
        plot(data.x); grid minor;
        title({['Mean = ' num2str(mean(data.x)) ', STD = ' num2str(std(data.x)) ];});
    elseif strcmpi(opt.dataPreprocessMode,'Data Standardization')
        data.x = DataStandardization(data.seriesdata);
        figure('Name','NormilizedInputData','NumberTitle','off');
        plot(data.x); grid minor;
        title({['Mean = ' num2str(mean(data.x)) ', STD = ' num2str(std(data.x)) ];});
    end
    
else
    warning(['In order to train network, please load data.' ...
        'Input data should be an excel file with only one column!']);
    disp('Operation Cancel.');
    data.isDataRead = false;
end
end
function data = PreInput(data)
if iscell(data)
    for i=1:size(data,1)
        for j=1:size(data,2)
            if strcmpi(data{i,j},'#NULL!')
                tempVars(i,j) = NaN; 
            else
                tempVars(i,j) = str2num(data{i,j});   
            end
        end
    end
    data = tempVars;
end
end
function vars = DataStandardization(data)
for i=1:size(data,2)
    x.mu(1,i)   = mean(data(:,i),'omitnan');
    x.sig(1,i)  = std (data(:,i),'omitnan');
    vars(:,i) = (data(:,i) - x.mu(1,i))./ x.sig(1,i);
end
end
function vars = DataNormalization(data)
for i=1:size(data,2)
    vars(:,i) = (data(:,i) -min(data(:,i)))./ (max(data(:,i))-min(data(:,i)));
end
end
function [opt,data] = PrepareData(opt,data)
% 为时序网络准备延迟
data = CreateTimeSeriesData(opt,data);
% 将数据分为测试数据和训练数据
data = dataPartitioning(opt,data);
% LSTM data form LSTM数据形式
data = LSTMInput(data);
end
% ----运行LSTM网络参数的贝叶斯优化超参数
function [opt,data] = OptimizeLSTM(opt,data)
if opt.isDispOptimizationLog
    isLog = 2;
else
    isLog = 0;
end
if opt.isUseOptimizer
    opt.ObjFcn  = ObjFcn(opt,data);
    BayesObject = bayesopt(opt.ObjFcn,opt.optimVars, ...
        'MaxTime',opt.MaxOptimizationTime, ...
        'IsObjectiveDeterministic',false, ...
        'MaxObjectiveEvaluations',opt.MaxItrationNumber,...
        'Verbose',isLog,...
        'UseParallel',false);
end
end
function ObjFcn = ObjFcn(opt,data)
ObjFcn = @CostFunction;
function [valError,cons,fileName] = CostFunction(optVars)
inputSize    = size(data.X,1);
outputMode   = 'last';
numResponses = 1;
dropoutVal   = .5;
if optVars.isUseBiLSTMLayer == 2
    optVars.isUseBiLSTMLayer = 0;
end
if opt.isUseDropoutLayer % 如果dropout layer为真
    if optVars.NumOfLayer ==1
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer==2
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer ==3
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer==4
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                dropoutLayer(dropoutVal)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                dropoutLayer(dropoutVal)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    end
else
    % 如果dropout layer为false
    if optVars.NumOfLayer ==1
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer ==2
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                lstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                lstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer ==3
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    elseif optVars.NumOfLayer ==4
        if optVars.isUseBiLSTMLayer
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        else
            opt.layers = [ ...
                sequenceInputLayer(inputSize)
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode','sequence')
                bilstmLayer(optVars.NumOfUnits,'OutputMode',outputMode)
                fullyConnectedLayer(numResponses)
                regressionLayer];
        end
    end
end
miniBatchSize    = opt.miniBatchSize;
maxEpochs        = opt.maxEpochs;
trainingProgress = opt.trainingProgress;
executionEnvironment = opt.executionEnvironment;
validationFrequency  = floor(numel(data.XTr)/miniBatchSize);

opt.opts = trainingOptions(opt.LR, ...
    'MaxEpochs',maxEpochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',optVars.InitialLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'L2Regularization',optVars.L2Regularization, ...
    'Verbose',1, ...
    'MiniBatchSize',miniBatchSize,...
    'ExecutionEnvironment',executionEnvironment,...
    'ValidationData',{data.XVl,data.YVl}, ...
    'ValidationFrequency',validationFrequency,....
    'Plots',trainingProgress);
disp('LSTM architect successfully created.');
try
    data.BiLSTM.Net = trainNetwork(data.XTr,data.YTr,opt.layers,opt.opts);
    disp('LSTM Netwwork successfully trained.');
    data.IsNetTrainSuccess =true;
catch me
    disp('Error on Training LSTM Network');
    data.IsNetTrainSuccess = false;
    return;
end
close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))
predict(data.BiLSTM.Net,data.XVl,'MiniBatchSize',opt.miniBatchSize);
valError = mse(predict(data.BiLSTM.Net,data.XVl,'MiniBatchSize',opt.miniBatchSize)-data.YVl);
Net  = data.BiLSTM.Net;
Opts = opt.opts;
fieldName = ['ValidationError' strrep(num2str(valError),'.','_')];
if ismember('OptimizedParams',evalin('base','who'))
    OptimizedParams =  evalin('base', 'OptimizedParams');
    OptimizedParams.(fieldName).Net  = Net;
    OptimizedParams.(fieldName).Opts = Opts;
    assignin('base','OptimizedParams',OptimizedParams);
else
    OptimizedParams.(fieldName).Net  = Net;
    OptimizedParams.(fieldName).Opts = Opts;
    assignin('base','OptimizedParams',OptimizedParams);
end
fileName = num2str(valError) + ".mat";
if opt.isSaveOptimizedValue
    save(fileName,'Net','valError','Opts')
end
cons = [];
end
end
function data = CreateTimeSeriesData(opt,data)
Delays = opt.Delays;
x = data.x';
T = size(x,2);
MaxDelay = max(Delays);
Range = MaxDelay+1:T;
X= [];
for d = Delays
    X=[X; x(:,Range-d)];
end
Y = x(:,Range);
data.X  = X;
data.Y  = Y;
end
% 划分输入数据
function data = dataPartitioning(opt,data)
data.XTr   = [];
data.YTr   = [];
data.XTs   = [];
data.YTs   = [];
numTrSample = round(opt.trPercentage*size(data.X,2));
data.XTr   = data.X(:,1:numTrSample);
data.YTr   = data.Y(:,1:numTrSample);
data.XTs   = data.X(:,numTrSample+1:end);
data.YTs   = data.Y(:,numTrSample+1:end);
disp(['Time Series data divided to ' num2str(opt.trPercentage*100) '% Train data and ' num2str((1-opt.trPercentage)*100) '% Test data']);
end
% 准备LSTM网络的输入数据。
function data = LSTMInput(data)
for i=1:size(data.XTr,2)
    XTr{i,1} = data.XTr(:,i);
    YTr(i,1) = data.YTr(:,i);
end
for i=1:size(data.XTs,2)
    XTs{i,1} =  data.XTs(:,i);
    YTs(i,1) =  data.YTs(:,i);
end
data.XTr   = XTr;
data.YTr   = YTr;
data.XTs   = XTs;
data.YTs   = YTs;
data.XVl   = XTs;
data.YVl   = YTs;
disp('Time Series data prepared as suitable LSTM Input data.');
end
function [opt,data] = EvaluationData(opt,data)
if opt.isUseOptimizer
    OptimizedParams =  evalin('base', 'OptimizedParams');
    % find best Net
    [valBest,indxBest] = sort(str2double(extractAfter(strrep(fieldnames(OptimizedParams),'_','.'),'Error')));
    data.BiLSTM.Net = OptimizedParams.(['ValidationError' strrep(num2str(valBest(1)),'.','_')]).Net;
    if opt.isSaveBestOptimizedValue
        fileName = ['BestNet ' num2str(valBest(1)) ' ' char(datetime('now','Format','yyyy.MM.dd HH.mm')) '.mat'];
        Net = data.BiLSTM.Net;
        save(fileName,'Net')
    end
elseif ~opt.isUseOptimizer
    [chosenfile,chosendirectory] = uigetfile({'*.mat'},...
    'Select Net File','BestNet.mat');
    if chosenfile==0
        error('Please Select saved Network File or set isUseOptimizer: true');
    end
    filePath = [chosendirectory chosenfile];
    Net = load(filePath);
    data.BiLSTM.Net = Net.Net;
end
data.BiLSTM.TrainOutputs = deNorm(data.seriesdata,predict(data.BiLSTM.Net,data.XTr,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
data.BiLSTM.TrainTargets = deNorm(data.seriesdata,data.YTr,opt.dataPreprocessMode);
data.BiLSTM.TestOutputs  = deNorm(data.seriesdata,predict(data.BiLSTM.Net,data.XTs,'MiniBatchSize',opt.miniBatchSize),opt.dataPreprocessMode);
data.BiLSTM.TestTargets  = deNorm(data.seriesdata,data.YTs,opt.dataPreprocessMode);
data.BiLSTM.AllDataTargets = [data.BiLSTM.TrainTargets data.BiLSTM.TestTargets];
data.BiLSTM.AllDataOutputs = [data.BiLSTM.TrainOutputs data.BiLSTM.TestOutputs];
data = PlotResults(data,'Tr',...
    data.BiLSTM.TrainOutputs, ...
    data.BiLSTM.TrainTargets);
data = plotReg(data,'Tr',data.BiLSTM.TrainTargets,data.BiLSTM.TrainOutputs);
data = PlotResults(data,'Ts',....
    data.BiLSTM.TestOutputs, ...
    data.BiLSTM.TestTargets);
data = plotReg(data,'Ts',data.BiLSTM.TestTargets,data.BiLSTM.TestOutputs);
data = PlotResults(data,'All',...
    data.BiLSTM.AllDataOutputs, ...
    data.BiLSTM.AllDataTargets);
data = plotReg(data,'All',data.BiLSTM.AllDataTargets,data.BiLSTM.AllDataOutputs);
disp('Bi-LSTM network performance evaluated.');
end
function vars = deNorm(data,stdData,deNormMode)
if iscell(stdData(1,1))
    for i=1:size(stdData,1)
        tmp(i,:) = stdData{i,1}';
    end
    stdData = tmp;
end
if strcmpi(deNormMode,'Data Normalization')
    for i=1:size(data,2)
        vars(:,i) = (stdData(:,i).*(max(data(:,i))-min(data(:,i)))) + min(data(:,i));
    end
    vars = vars';
elseif strcmpi(deNormMode,'Data Standardization')
    for i=1:size(data,2)
        x.mu(1,i)   = mean(data(:,i),'omitnan');
        x.sig(1,i)  = std (data(:,i),'omitnan');
        vars(:,i) = ((stdData(:,i).* x.sig(1,i))+ x.mu(1,i));
    end
    vars = vars';
else
    vars = stdData';
    return;
end
end
function data = PlotResults(data,firstTitle,Outputs,Targets)
Errors = Targets - Outputs;
MSE   = mean(Errors.^2);
RMSE  = sqrt(MSE);
NRMSE = RMSE/mean(Targets);
ErrorMean = mean(Errors);
ErrorStd  = std(Errors);
rankCorre = RankCorre(Targets,Outputs);
if strcmpi(firstTitle,'tr')
    Disp1Name = 'OutputGraphEvaluation_TrainData';
    Disp2Name = 'ErrorEvaluation_TrainData';
    Disp3Name = 'ErrorHistogram_TrainData';
elseif strcmpi(firstTitle,'ts')
    Disp1Name = 'OutputGraphEvaluation_TestData';
    Disp2Name = 'ErrorEvaluation_TestData';
    Disp3Name = 'ErrorHistogram_TestData';
elseif strcmpi(firstTitle,'all')
    Disp1Name = 'OutputGraphEvaluation_ALLData';
    Disp2Name = 'ErrorEvaluation_ALLData';
    Disp3Name = 'ErrorHistogram_AllData';
end
figure('Name',Disp1Name,'NumberTitle','off');
plot(1:length(Targets),Targets,...
    1:length(Outputs),Outputs);grid minor
legend('Targets','Outputs','Location','best') ;
title(['Rank Correlation = ' num2str(rankCorre)]);
figure('Name',Disp2Name,'NumberTitle','off');
plot(Errors);grid minor
title({['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)...
    ' NRMSE = ' num2str(NRMSE)] ;});
xlabel(['Error Per Sample']);
figure('Name',Disp3Name,'NumberTitle','off');
histogram(Errors);grid minor
title(['Error Mean = ' num2str(ErrorMean) ', Error StD = ' num2str(ErrorStd)]);
xlabel(['Error Histogram']);
if strcmpi(firstTitle,'tr')
    data.Err.MSETr = MSE;
    data.Err.STDTr = ErrorStd;
    data.Err.NRMSETr     = NRMSE;
    data.Err.rankCorreTr = rankCorre;
elseif strcmpi(firstTitle,'ts')
    data.Err.MSETs = MSE;
    data.Err.STDTs = ErrorStd;
    data.Err.NRMSETs     = NRMSE;
    data.Err.rankCorreTs = rankCorre;
elseif strcmpi(firstTitle,'all')
    data.Err.MSEAll = MSE;
    data.Err.STDAll = ErrorStd;
    data.Err.NRMSEAll     = NRMSE;
    data.Err.rankCorreAll = rankCorre;
end
end
% 找出网络输出和实际数据之间的等级相关性
function [r]=RankCorre(x,y)
x=x';
y=y';
% 查找数据长度
N = length(x);
% 得到x的秩
R = crank(x)';
for i=1:size(y,2)
    S = crank(y(:,i))';
    % 计算相关系数
    r(i) = 1-6*sum((R-S).^2)/N/(N^2-1); 
end
end
function r=crank(x)
u = unique(x);
[~,z1] = sort(x);
[~,z2] = sort(z1);
r = (1:length(x))';
r=r(z2);
for i=1:length(u)
    s=find(u(i)==x);
    r(s,1) = mean(r(s));
end
end
% 绘制产量与实际值的回归线
function data = plotReg(data,Title,Targets,Outputs)
if strcmpi(Title,'tr')
    DispName = 'RegressionGraphEvaluation_TrainData';
elseif strcmpi(Title,'ts')
    DispName = 'RegressionGraphEvaluation_TestData';
elseif strcmpi(Title,'all')
    DispName = 'RegressionGraphEvaluation_ALLData';
end
figure('Name',DispName,'NumberTitle','off');
x = Targets';
y = Outputs';
format long
b1 = x\y;
yCalc1 = b1*x;
scatter(x,y,'MarkerEdgeColor',[0 0.4470 0.7410],'LineWidth',.7);
hold('on');
plot(x,yCalc1,'Color',[0.8500 0.3250 0.0980]);
xlabel('Prediction');
ylabel('Target');
grid minor
% xgrid = 'on';
% disp.YGrid = 'on';
X = [ones(length(x),1) x];
b = X\y;
yCalc2 = X*b;
plot(x,yCalc2,'-.','MarkerSize',4,"LineWidth",.1,'Color',[0.9290 0.6940 0.1250])
legend('Data','Fit','Y=T','Location','best');
%
Rsq2 = 1 -  sum((y - yCalc1).^2)/sum((y - mean(y)).^2);
if strcmpi(Title,'tr')
    data.Err.RSqur_Tr = Rsq2;
    title(['Train Data, R^2 = ' num2str(Rsq2)]);
elseif strcmpi(Title,'ts')
    data.Err.RSqur_Ts = Rsq2;
    title(['Test Data, R^2 = ' num2str(Rsq2)]);
elseif strcmpi(Title,'all')
    data.Err.RSqur_All = Rsq2;
    title(['All Data, R^2 = ' num2str(Rsq2)]);
end
end

