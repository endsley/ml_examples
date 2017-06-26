
% find points that are within epsilon
function proximity_reduction(A, epsilon)
	[row,col] = find(A == 1);
	points = [row,col];

	points
end
