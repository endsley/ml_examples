
function mag = obj_fun( x1, x2, q,W, Z, L1, L2)
	mag = -exp(-x1'*W*W'*x1) - 2*exp(-x2'*W*W'*x2) + trace(L1*(W'*Z-eye(q))) + trace(L2*(W-Z)) + sum(sum((Z'*W-eye(q)).*(Z'*W-eye(q)))) + sum(sum((W-Z).*(W-Z)));
end
