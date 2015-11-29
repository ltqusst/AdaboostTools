function [y features] = LoadSample_zq()
    %特征维数
    feature_cnt = 116;
    
    %读入全部样本
    fmt = [];
    for i=1:feature_cnt 
        fmt = [fmt '%g'];
    end
    
    fid=fopen('K:\plate_sample\0_P.txt', 'r');
    %fgets(fid); %跳过样本个数
    features_pos = fscanf(fid, fmt, [feature_cnt inf])';
    y_pos = ones(size(features_pos, 1),1);
    fclose(fid);
    
    fid=fopen('K:\plate_sample\0_N.txt', 'r');
    %fgets(fid); %跳过样本个数
    features_neg = fscanf(fid, fmt, [feature_cnt inf])';
    y_neg = -ones(size(features_neg, 1),1);
    fclose(fid);
    
    y=[y_pos; y_neg];
    features=[features_pos; features_neg];
end
