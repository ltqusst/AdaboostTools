function genBoost()
%��������Ҫ��һ�����������ļ���ÿ��������ʽΪ:
% class(1 or -1)�� feature1, feature2, ..., featureN 
% 
%ѵ��һ��ָ��ά����gentle Adaboost������
%     ÿ������������3������(th,L,R)��ɣ�����ʽΪ:
%                fm(x) = x>th? L:R;


%{
%����ά��
feature_cnt = 10;
weakleaner_cnt = 4;
sample_file = 'samples.txt';

%����ȫ������
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
[y features] = LoadSample_zq(); %��������
GenBoost = trainGenBoost(y, features, weakleaner_cnt, th_try_count, show_info); %ѵ��

%ʹ��Boost������ǿ���������ʶ��
class = classifyGenBoost(GenBoost, features);
miss_classify = abs(class - y)/2;
error_rate = sum(miss_classify)/(size(y,1));
fprintf('error_rate = %g(%d/%d)\r\n', error_rate, sum(miss_classify), (size(y,1)));
end

