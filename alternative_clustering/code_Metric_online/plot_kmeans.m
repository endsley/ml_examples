
function plot_kmeans(X, opt_idx)
	figure(1);hold on;
	for m = 1:length(X)
		if opt_idx(m) == 1
			plot(X(m,1),X(m,2),'ro');
		else
			plot(X(m,1),X(m,2),'bo');
		end
	end
end
