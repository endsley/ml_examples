%	Compute the function of Tr(11^T (Z^T*W-I).*(Z^T*W-I))


function F = compute_J(W, Z)
	F = sum(sum(((W - Z).*(W - Z))));
end

%function F = compute_J(W, Z, L1,L2,q)
%	F = trace(L2*(W - Z));
%end

%function F = compute_J(W, Z, L1,L2,q)
%	F = trace(L1*(W'*Z-eye(q)));
%end

%function F = compute_J(W, Z, x1, x2)
%	F = exp(-x1'*W*W'*x1) + 2*exp(-x2'*W*W'*x2);
%end
