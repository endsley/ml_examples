
function plot_kmeans(X, opt_idx, colormap)
	hold on;
	for m = 1:length(X)
		c = colormap(opt_idx(m),:);
		plot(X(m,1),X(m,2),'x', 'Color', c);
	end
	hold off;
end
