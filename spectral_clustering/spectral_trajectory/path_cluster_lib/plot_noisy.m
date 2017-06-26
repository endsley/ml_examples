
function plot_noisy(y,color, original_view)
	l = size(y,2);
	x = [1:size(y,1)]';
	hold on;
	for m = 2:l
		y_column = y(:,m);
		if(original_view == 1)
			plot(x,y_column,'k');
		else
			plot(x,y_column,color);
		end
	end

	if(original_view == 1)
		plot(x,y(:,1) ,'k')
	else
		plot(x,y(:,1) ,'k', 'LineWidth',2)
	end
end
