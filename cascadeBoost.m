function cascadeBoost()
% ����ǿ��������ͨ������ǿBoost����������ֵ0��ʹ������ͨ���ʴﵽָ��Ҫ��
% 
% 
weakleaner_cnt = 10;
th_try_count = 20;
show_info = true;
[y features] = LoadSample_ltq();
y_org = y;
features_org = features;

stage_weakleaner_cnts=[16  30  60  150]
for stage = 1:size(stage_weakleaner_cnts,2)
    %train stage
    GenBoost{stage} = trainGenBoost(y, features, stage_weakleaner_cnts(stage), 20, false);
    %decrease th to meet pass rate
    [BoostTh(stage) pos_pass_rate neg_kill_rate] = AdjustBoostTh(GenBoost{stage}, y, features, 0.9995);
    
    fprintf('[%d] th=%g, ������ͨ����=%g%%, ������ɾ����=%g%%\r\n', stage, ...
                BoostTh(stage), pos_pass_rate*100, neg_kill_rate*100);
    
    %next stage only use passed samples
    class = classifyGenBoost(GenBoost{stage}, features, BoostTh(stage));
    y(class == -1) = [];
    features(class == -1,:) = [];
end

save('CascadeGenBoost.dat', 'GenBoost', 'BoostTh');
    
pos_passed = sum(y == 1)/sum(y_org == 1);
neg_passed = sum(y == -1)/sum(y_org == -1);
fprintf('final pass rate: ������ͨ����=%g%%, ������ͨ����=%g%%\r\n', pos_passed*100, neg_passed*100);
end

%������ֵʹ��������ͨ���ʴﵽָ��Ŀ��
function [BoostTh pos_pass_rate neg_kill_rate] = AdjustBoostTh(GenBoost, y, features, pos_pass_rate_target)
    BoostTh = 0.01;
    pos_pass_rate = 0;
    neg_kill_rate = 0;
    
    %�������
    weakleaner_cnt = size(GenBoost,2);
    f = zeros(size(features(:,1)));
    for i=1:weakleaner_cnt
        j = GenBoost{i}.j;
        th = GenBoost{i}.th;
        A = GenBoost{i}.A;
        B = GenBoost{i}.B;
        %�ۼ�ÿ�����������б���
        fmi = double(features(:, j) < th)*(A-B)+B; % xi<th? A:B
        f = f + fmi;
    end
    
    while(pos_pass_rate < pos_pass_rate_target)
        %ֻҪ������ͨ���ʲ���꣬���Թ̶�����������ֵ
        BoostTh = BoostTh - 0.01;
        class = double(f > BoostTh)*2 - 1; % f>0?1:-1
        %class = classifyGenBoost(GenBoost, features, BoostTh);
        pos_pass_rate = sum(double((class == 1) & (y == 1)))/sum(double(y == 1));
        neg_kill_rate = sum(double((class == -1) & (y == -1)))/sum(double(y == -1));
    end
end
