function class = classifyGenBoost(GenBoost, features, BoostTh)
%features中每行就是一个样本
if nargin < 3
    BoostTh = 0;
end

weakleaner_cnt = size(GenBoost,2);
f = zeros(size(features(:,1)));
for i=1:weakleaner_cnt
    node = GenBoost{i};
    h = h_node(node,features);
    f = f + h(:);
end
class = double(f > BoostTh)*2 - 1; % f>0?1:-1
end



function h = h_node(node, features)
    mask0 = (features(:, node.j) < node.th);
    mask1 = ~mask0;
    if(node.depth == 1)
        h = double(mask0) .* node.a0 + double(mask1) .* node.a1;
    else
        %call next level sub-tree
        h(mask0) = h_node(node.child0, features(mask0,:));
        h(mask1) = h_node(node.child1, features(mask1,:));
    end
end
