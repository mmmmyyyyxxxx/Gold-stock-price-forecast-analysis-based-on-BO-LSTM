%% 导入电子表格中的数据
% 用于从以下电子表格导入数据的脚本:
%
%    工作簿: D:\各类竞赛文件\大数据竞赛\数据分析实践赛\尝试\data.xlsx
%    工作表: Sheet1
%
% 由 MATLAB 于 2023-12-17 21:09:12 自动生成

%% 设置导入选项并导入数据
opts = spreadsheetImportOptions("NumVariables", 7);

% 指定工作表和范围
opts.Sheet = "Sheet1";
opts.DataRange = "A2:G623";

% 指定列名称和类型
opts.VariableNames = ["Data", "Open", "High", "Low", "Close", "ChangePips", "Change"];
opts.VariableTypes = ["datetime", "double", "double", "double", "double", "double", "double"];

% 指定变量属性
opts = setvaropts(opts, "Data", "InputFormat", "");

% 导入数据
data1 = readtable("D:\各类竞赛文件\大数据竞赛\数据分析实践赛\尝试\data.xlsx", opts, "UseExcel", false);


%% 清除临时变量
clear opts
data=table2timetable(data1)