
function curvature_function = get_menger_curvature(data)
	inc = 5;
	curvature_function = [];
	filter_len = 10;

	for m = 1:length(data)
		if (m-inc) < 1 % avoid the first points due to edge conditions
			curvature_function = [curvature_function, 0];
		elseif (m+inc) > length(data)
			curvature_function = [curvature_function, 0];
		else
			pq = sqrt(inc^2 + (data(m - inc) - data(m))^2);
			qr = sqrt(inc^2 + (data(m + inc) - data(m))^2);
			rp = sqrt((inc*2)^2 + (data(m + inc) - data(m - inc))^2);
			A = 4*pq^2*qr^2;
			B = pq^2 + qr^2 - rp^2;
			DC = sqrt(A-B^2)/(pq*qr*rp);
			curvature_function = [curvature_function, DC];
		end
	end
	curvature_function(floor(length(curvature_function)/2):end) = 0;
	curvature_function = curvature_function/max(curvature_function);

%	DD = data;
%	DD(1:2) = DD(3);	% prevent the first spike
%	vel = [diff(DD),0];
%	vel(floor(length(vel)/2):end) = 0;
%	vel = detrend(vel,6);


	%vel = vel/max(vel);
%	curvature_function = vel + 2*curvature_function;
	%curvature_function = vel;
%[DD ;vel]
	%curvature_function = detrend(curvature_function,6);

	%[b,a]= butter ( 2, 0.1 );
	%curvature_function = filter(b,a, curvature_function);

	%fir_filter = fir1(filter_len,0.1);
	%curvature_function = conv(fir_filter,curvature_function);
	%curvature_function = curvature_function(floor(filter_len/2)+1:end - floor(filter_len/2)+0);

	curvature_function = (curvature_function/max(curvature_function));
%	curvature_function = [0, 0, curvature_function(1:end-2)];
end
