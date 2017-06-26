
function [cutoff_point, filtered_curv] = get_peaks(x, curv)
	gap = ceil(length(x)*0.03);
	%gap = 7;

	filtered_curv = x;
	filtered_curv(filtered_curv < 1.5*std(filtered_curv)) = 0;
	cutoff_point = gap;

	if(sum(filtered_curv) == 0)
		filtered_curv = x;
		filtered_curv(filtered_curv < std(filtered_curv)) = 0;
	end

	if(sum(filtered_curv) == 0)
		filtered_curv = x;
		filtered_curv(filtered_curv < 0.6*std(filtered_curv)) = 0;
	end

	curv_sum = sum(filtered_curv);

	vel = [0,diff(filtered_curv)];
	vel( vel > 0 ) = 1;
	vel( vel < 0 ) = -1;
	
	acc = [0,diff(vel)];
	acc = acc == -2;

	[p, idx] = find(acc);

	if(length(idx) == 1)
		cutoff_point = idx(1);
	else 
		if( sum(idx > gap) != 0 )
			idx = idx(idx > gap);
		end

		if(length(idx) == 1)
			cutoff_point = idx(1);
		else
			angle_1 = get_curvature_angle(idx(1), curv);
			angle_2 = get_curvature_angle(idx(2), curv);

			%% residual points
			%left_p = idx(1) - 5;
			%right_p = idx(1) + 5;
			%y = [curv(left_p); curv(right_p)];

			%A = [[left_p;right_p], [1;1]];
			%[q r] = qr(A);
			%coef = r\(q'*y);

			%pp = [[left_p:right_p]', ones(length([left_p:right_p]),1)]*coef;
			%p_error_1 = sum(abs(pp - curv(left_p:right_p)'))^2/length([left_p:right_p])
			%plot( left_p:right_p , pp,'b'); hold on;

			%%%%

			%left_p = idx(2) - 5;
			%right_p = idx(2) + 5;
			%y = [curv(left_p); curv(right_p)];

			%A = [[left_p;right_p], [1;1]];
			%[q r] = qr(A);
			%coef = r\(q'*y);

			%pp = [[left_p:right_p]', ones(length([left_p:right_p]),1)]*coef;
			%p_error_1 = sum(abs(pp - curv(left_p:right_p)'))^2/length([left_p:right_p])
			%plot( left_p:right_p , pp,'b'); hold on;



			if(angle_1 < 0.7)
				cutoff_point = idx(2);
			elseif(abs(idx(2) - idx(1)) < 2*gap)
				cutoff_point = max([idx(1),idx(2)]);
			else
				cutoff_point = idx(1);
			end



		end
	end
'--------------------------------'
%	for(m = 1:length(idx))
%		if(idx(m) < gap)
%			peak_y(idx(m)) = 0;
%		end
%	end


%		up = sum(peak_y(idx(m)-5+1:idx(m)) > 0) == gap
%%peak_y(idx(m):idx(m)+gap-1)
%%		down = sum(peak_y(idx(m):idx(m)+gap-1) < 0) == gap
%%		is_peak = up*down;
%
%		
%		%if(is_peak == 0)
%		%	peak_y(idx(m)) = 0;
%		%end
%	end

%	[vel; peak_y]

	%peak_y = peak_y/max(peak_y);

%	[vel;peaks]
%	gap
end
