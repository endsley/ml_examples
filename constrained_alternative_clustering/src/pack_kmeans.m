
function [opt_idx , opt_C, sumd] = pack_kmeans(X,k, repeat_time)
	opt_error_d = -1;
	opt_idx = 0;
	opt_C = 0;

	for m = 1:repeat_time
		[idx,C,sumd] = kmeans(X,k);
		error_d =norm(sumd);
		if(opt_error_d == -1)
			opt_error_d = error_d;
			opt_idx = idx;
			opt_C = C;
		end
	
		if(error_d < opt_error_d)
			opt_idx = idx;
			opt_error_d = error_d;
			opt_C = C;
		end
	end
end
