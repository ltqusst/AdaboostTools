function [y features] = LoadSample_ltq()
    %特征维数
    feature_cnt = 116+5+5+1;
    
    %读入全部样本
    fmt = ['%g'];
    for i=1:feature_cnt 
        fmt = [fmt '%g'];
    end
    
    fid=fopen('H:\projdata\ANPR\sample.txt', 'r');
    %fgets(fid); %跳过样本个数
    sample = fscanf(fid, fmt, [feature_cnt + 1 inf])';
    fclose(fid);
        
    y=sample(:,1);
    features=sample(:,2:feature_cnt+1);
end
