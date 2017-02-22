
%	ordering = 'ascend' or 'descend'
function [eig_Vector, eig_Value] = sorted_eig(X, ordering)
	[V,D] = eig(X);
	[eig_Value,o] = sort(diag(D), ordering);
	eig_Vector = V(:,o);
end
