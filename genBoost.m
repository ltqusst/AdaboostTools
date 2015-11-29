function genBoost()
%本函数是要求一个输入样本文件，每个样本格式为:
% class(1 or -1)， feature1, feature2, ..., featureN 
% 
%训练一个指定维数的gentle Adaboost分类器
%     每个弱分类器由3个参数(th,L,R)组成，其形式为:
%                fm(x) = x>th? L:R;


%{
%特征维数
feature_cnt = 10;
weakleaner_cnt = 4;
sample_file = 'samples.txt';

%读入全部样本
fmt = ['%g:'];
for i=1:feature_cnt 
    fmt = [fmt '%g,'];
end
fid=fopen(sample_file, 'r');
samples = fscanf(fid, fmt, [feature_cnt + 1 inf])';
fclose(fid);

y = samples(:,1);
features = samples(:,2:feature_cnt+1);
%}

weakleaner_cnt = 50;
th_try_count = 20;
show_info = true;
[y features] = LoadSample_zq(); %加载样本
GenBoost = trainGenBoost(y, features, weakleaner_cnt, th_try_count, show_info); %训练

%使用Boost给出的强分类器完成识别
class = classifyGenBoost(GenBoost, features);
miss_classify = abs(class - y)/2;
error_rate = sum(miss_classify)/(size(y,1));
fprintf('error_rate = %g(%d/%d)\r\n', error_rate, sum(miss_classify), (size(y,1)));
end

