function [y features] = LoadSample_ltq()
    %����ά��
    feature_cnt = 116+5+5+1;
    
    %����ȫ������
    fmt = ['%g'];
    for i=1:feature_cnt 
        fmt = [fmt '%g'];
    end
    
    fid=fopen('H:\projdata\ANPR\sample.txt', 'r');
    %fgets(fid); %������������
    sample = fscanf(fid, fmt, [feature_cnt + 1 inf])';
    fclose(fid);
        
    y=sample(:,1);
    features=sample(:,2:feature_cnt+1);
end
