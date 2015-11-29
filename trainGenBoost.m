function GenBoost = trainGenBoost(y, features, weakleaner_cnt, show_info, CARTdepth)
% train Adaboost with CART of specified depth as weaklearner
%       each node of CART:
%                h(x) = x<th? Left_Node(a0) : Right_Node(a1);
[sample_cnt feature_cnt] = size(y);

%样本权值初始化
w = ones(sample_cnt, 1);
w = w/sum(w);
%Boost
GenBoost = [];
for i=1:weakleaner_cnt

    %寻找最佳弱分类器fm
    node = trainNode(features, y, w, CARTdepth);

    %记录最佳弱分类器
    GenBoost{i} = node;

    %权值更新
    w = w.*exp(-y.*node.h);
    %权值归一化
    w = w/sum(w);
    if show_info
        class = classifyGenBoost(GenBoost, features);
        miss_classify = abs(class - y)/2;
        error_rate = sum(miss_classify)/(size(y,1));
        fprintf('[feat%d<%g\t a0=%f\t a1=%f]\t error_rate = %g(%d/%d)\n', ...
                                node.j - 1, ...
                                node.th, ...
                                node.a0, ...
                                node.a1, ...
                                error_rate, sum(miss_classify), (size(y,1)));
    end
end
end


function node = trainNode(features, y, w, depth)
    node = []; 
    node.err = inf;
    
    wy = w.*y;
    swy = sum(wy); 
    sw = sum(w);
    feature_cnt = size(features, 2);
    
    for j=1:feature_cnt
        best_j = findBestThreshold(features(:, j),  y, wy, w, swy, sw);
        if(best_j.err < node.err)
            node = best_j;
            node.j = j;
        end
    end
    node.depth = depth;
    node.train_cnt = size(features, 1);
    depth = depth - 1;
    
    if(depth > 0)
        mask0 = (features(:, node.j) < node.th);
        mask1 = (~mask0);
        node.child0 = trainNode(features(mask0,:), y(mask0), w(mask0), depth);
        node.child1 = trainNode(features(mask1,:), y(mask1), w(mask1), depth);
        if(~isfield(node.child0,'h') || ~isfield(node.child1,'h'))
            % some child only got single class sample, train is
            % over-fitting
            node.depth = 1;
        else
            node.h(mask0) = node.child0.h;
            node.h(mask1) = node.child1.h;
        end
    end
end

%寻找第j个弱分类器的最佳阈值th,应该是使得分类错误最小的th
%    err=sum(wi.*(yi-fm(xi))^2);
%    fm(xi) = xi<th?A:B
%    A = sum_for_xi<th(wi.*yi)
function node = findBestThreshold(features_cur, y, wy, w, swy, sw)
    node.err = inf;
    %因为输入数据的分布并非均匀分布，因此分段也不能是均匀分段的
    %最为合理的做法是，逐个尝试全部阈值
    for i=1:size(features_cur,1)
        th = features_cur(i);
        stump_left_mask = double(features_cur < th);
        stump_right_mask = 1 - stump_left_mask;
        
        % sum(weight(y=1) - weight(y=-1)) for partition0
        wy0 = sum(stump_left_mask.*wy);
        % sum(weight(y=1) - weight(y=-1)) for partition1
        wy1 = swy-wy0;
        
        % sum(weight(y=1) + weight(y=-1)) for partition0
        w0 = sum(stump_left_mask.*w);
        % sum(weight(y=1) + weight(y=-1)) for partition1
        w1 = sw - w0;
        
        if w0 == 0 || w1 == 0
            continue;
        end
        
        a0 = wy0/w0;
        a1 = wy1/w1;
        %计算least-square错误率
        h = (stump_left_mask.*a0) + (stump_right_mask.*a1);
        err = sum(w.*(y - h).^2);
        if err < node.err
            node.err = err;
            node.th = th;
            node.a0 = a0;
            node.a1 = a1;
            node.h = h;
            node.j = 0;
        end
    end
end
