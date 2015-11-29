
N = 2000;
features=rand(2*N, 2)*100;

%线性分类可以很好的处理
%y=(features(:,1) + 20> features(:,2))*2-1;

%圆形分类也可以很好的处理
%y=(((features(:,1)-50).^2 + (features(:,2)-50).^2)<20^2)*2-1;
%y=((100./(features(:,1)-50) + 100./(features(:,2)-50))>1)*2-1;

%异或形式无法处理
%y=(((features(:,1)-50).*(features(:,2)-50) - (features(:,2)-50).^2)>0)*2-1;
%y=((features(:,1)-50).*(features(:,2)-50) > 0)*2-1;

%多类聚类
y = (((features(:,1)-80).^2 + (features(:,2)-80).^2) < 100) | (((features(:,1)-30).^2 + (features(:,2)-30).^2) < 100) | (((features(:,1)-20).^2 + (features(:,2)-60).^2) < 100);
y = y*2-1;

%增加交叉项可以很好处理异或(但是同时也必须增加训练的强度)
%features = [features features(:,1).*features(:,2)];

%features = [features (features(:,1))./(features(:,2)-50)];
%features = [features (features(:,1))./(features(:,2)-50)];

train_features = features(1:N,:);
train_y =  y(1:N,:);
test_features = features(N:2*N,:);
test_y =  y(N:2*N,:);


weakleaner_cnt = 20;
show_info = true;
GenBoost = trainGenBoost(train_y, train_features, weakleaner_cnt, show_info,4); %训练

%使用Boost给出的强分类器完成识别

class = classifyGenBoost(GenBoost, test_features);
miss_classify = abs(class - test_y)/2;
error_rate = sum(miss_classify)/(size(test_y,1));
fprintf('error_rate = %g(%d/%d)\r\n', error_rate, sum(miss_classify), (size(test_y,1)));


subplot(1,2,1);
plot(train_features(train_y==1,1),train_features(train_y==1,2), 'xr','LineStyle','none');hold on;
plot(train_features(train_y==-1,1),train_features(train_y==-1,2), 'xb','LineStyle','none');
title('训练集合');hold off;

subplot(1,2,2);
plot(test_features(class==1,1),test_features(class==1,2), 'xr','LineStyle','none');hold on;
plot(test_features(class==-1,1),test_features(class==-1,2), 'xb','LineStyle','none');
title('测试集合');hold off;
