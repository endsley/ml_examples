
function out_dat = adjust_width(dat, max_len)
	out_dat = zeros(max_len,1);
	n = length(dat);
	
	dat
	for m = 1:max_len
		ratio1 = n*m/max_len;
		ratio2 = n*m/max_len - floor(n*m/max_len);
		upIdx = ceil(ratio1); 
		downIdx = floor(ratio1);

		if downIdx == 0
			downIdx = 1;
		end

		%[dat(downIdx) + ratio2*(dat(upIdx) - dat(downIdx)), upIdx, downIdx]
		%[upIdx, downIdx]
		%[ratio2*(dat(upIdx) - dat(downIdx)) + dat(downIdx), upIdx, downIdx]
	
		out_dat(m) = ratio2*(dat(upIdx) - dat(downIdx)) + dat(downIdx);
	end
end
