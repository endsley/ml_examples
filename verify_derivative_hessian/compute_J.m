% 

function F = compute_J(w,A1, A2)
	%F = -exp(-w'*A1*w) - 2*exp(-w'*A2*w);

	%F = -exp(-w'*A1*w);
	F = - 2*exp(-w'*A2*w);

	%F = 2*exp(-w.T.dot(A1).dot(w))*(A1 - 2*A1*w*w'*A1) + 4*exp(-w.T.dot(A2).dot(w))*(A2 - 2*A2*w*w'*A2);
end
