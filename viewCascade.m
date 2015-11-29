function viewCascade()
load 'CascadeGenBoost.dat' '-mat'

if 1
    fid = fopen('CascadeGenBoost.txt','w');
    fprintf(fid,'[%d]\r\n', size(GenBoost,2));
    for stage=1:size(GenBoost,2)
        GenBoostStage = GenBoost{stage};

        fprintf(fid,'(%d)\t%f\r\n', size(GenBoostStage,2), BoostTh(stage));
        for i=1:size(GenBoostStage,2)
            weakleaner = GenBoostStage{i};
            fprintf(fid,'\t%d,%f,%f,%f\r\n', ...
                        weakleaner.j, ...
                        weakleaner.th, ...
                        weakleaner.A, ...
                        weakleaner.B);
        end
    end
    fclose(fid);
end
%��ȡ����
[y features] = LoadSample_ltq();
sample_ind = 1:size(y,1);
y_org = y;
features_org = features;
%��ʶ��
for stage=1:size(GenBoost,2)
    class = classifyGenBoost(GenBoost{stage}, features, BoostTh(stage));
    %������ɸѡ���Ĳ������´�ʶ��
    sample_ind(class == -1) = [];
    y(class == -1) = [];
    features(class == -1,:) = [];
end
%ͳ������ͨ����
pos_passed = sum(y == 1)/sum(y_org == 1);
neg_passed = sum(y == -1)/sum(y_org == -1);
fprintf('final pass rate: ������ͨ����=%g%%(%d), ������ͨ����=%g%%(%d)\r\n', ...
        pos_passed*100, sum(y == 1), ...
        neg_passed*100, sum(y == -1));

    %ʣ��Ķ��Ǳ���Ϊ����������
    sample_ind((y == -1));

%{
feature_mask = zeros(1,116);
for stage=1:size(GenBoost,2)
    GenBoostStage = GenBoost{stage};
    f_list = [];
    for i=1:size(GenBoostStage,2)
        weakleaner = GenBoostStage{i};
        feature_mask(weakleaner.j) = feature_mask(weakleaner.j) + 1;
        f_list(i) = weakleaner.j;
    end
    
end
ind  = find(feature_mask > 0);
%}

end