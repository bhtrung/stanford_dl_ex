function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
hAct(1) = mat2cell(bsxfun(@rdivide, 1, 1 + exp(-stack{1}.W * data - repmat(stack{1}.b,1,size(data,2)))), size(stack{1}.W,1), size(data,2));
for i = 2:numHidden+1
    t = cell2mat(hAct(i-1));    
    hAct(i) = mat2cell(bsxfun(@rdivide, 1, 1 + exp(- stack{i}.W * t - repmat(stack{i}.b,1,size(t,2)))), size(stack{i}.W,1), size(t,2));
end
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
a = cell2mat(hAct(numHidden+1)); %num-classes x m  
e = bsxfun(@eq, labels, 1:ei.output_dim);     
b = bsxfun(@rdivide, exp(a), sum(exp(a'))');  
cost = sum(sum(e.*log(b')));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
gradStack = -(data * (e - b'));

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



