

%	Define some common constants 
	x1 = [1;0;1]; x2 = [3;2;1];
	d = 3; q = 2;

	Z = [1 0;0 1;0 0];
	W = [1 0;0 1;0 0];
	L1 = [1 0;0 2];
	L2 = [2 3 1;0 0 1];
	L3 = [2 0 1;0 0 1];

	alpha = 0.1;

%	Calculate W gradient
for p = 1:300

	lastF = 10000;
	for m = 1:200
		dW = (2*x1*x1'*W)*exp(-x1'*W*W'*x1) + 4*(x2*x2'*W)*exp(-x2'*W*W'*x2) + Z*L1 + L2' + L3' + 2*Z*(Z'*W-eye(q)) + 2*(W-Z);
		new_W = W - alpha*dW;
		F = obj_fun( x1, x2, q,new_W, Z, L1, L2);
		if (lastF - F) > lastF*0.001
			W = W - alpha*dW;
			lastF = F;
		else
			alpha = alpha*0.8;
	        if alpha < 0.0001
	            break;
	        end
		end
	end
	 
	
	
	alpha = 0.1;
	lastF = 10000;
	for m = 1:200
		dZ = +W*L1' - L2' + 2*W*(W'*Z-eye(q)) - 2*(W-Z);
		new_Z = Z - alpha*dZ;
		F = obj_fun( x1, x2, q,W, new_Z, L1, L2);
	
		if (lastF - F) > lastF*0.001
			Z = Z - alpha*dZ;
			lastF = F;
		else
			alpha = alpha*0.8;
	        if alpha < 0.0001
	            break;
	        end
		end
	
	end
	

		L = [L1, L2];
		D = [Z';eye(d)]*W - [zeros(q,d);eye(d)]*Z - [eye(q);zeros(d,q)];
		L = L + 0.05*D';
		L1 = L(:,1:q);
		L2 = L(:,q+1:q+d);
		F1 = -obj_fun( x1, x2, q,W, Z, L1, L2);

	if F1 > 2.4
		W'*Z
	end
%
end

WZ_estimate = W'*Z
F1 = -obj_fun( x1, x2, q,W, Z, L1, L2)




