
function [curvature_function, cutoff_point] = get_chieh_curvature(data)


%	inc = ceil(length(data)*0.02);
%	curvature_function = [];
%	cutoff_point = 10;
%
%	for m = 1:length(data)
%	%for m = 1:100
%		m
%		if (m-inc) < 1
%			curvature_function = [curvature_function, 0];
%		elseif ((m+inc) > length(data))
%			curvature_function = [curvature_function, 0];
%		else
%			part1 = data(m - inc:m)';
%			part2 = data(m:m + inc)';
%
%			%max_1 = max([part1 ; part2]);
%			%min_1 = min([part1 ; part2]);
%			%mid = (max_1 + min_1)/2;
%			%x1 = [min_1:(mid- min_1)/inc:mid]';
%			%x2 = [mid:(max_1-mid)/inc:max_1]';
%			
%			x1 = [0:inc]';
%			x2 = [inc+1:2*inc+1]';
%
%			A = [x1, ones(inc + 1,1)];
%			[q r] = qr(A);
%			vec1 = r\(q'*part1);
%
%			A = [x2, ones(inc + 1,1)];
%			[q r] = qr(A);
%			vec2 = r\(q'*part2);
%
%			error1 = norm([x1, ones(inc + 1,1)]*vec1(:) - part1)/length(part1);
%			error2 = norm([x2, ones(inc + 1,1)]*vec2(:) - part2)/length(part2);
%
%			A = [0, 1; 2*inc + 1, 1];
%			y = [data(m - inc);data(m + inc)];
%			[q r] = qr(A);
%			vec = r\(q'*y);
%
%			x = [0:2*inc+1]';
%			error_total = norm([x, ones(inc*2 + 2,1)]*vec(:) - [part1;part2])/length(part1);
%
%			error_ratio = error_total/(error1 + error2)
%			curvature_function = [curvature_function, error_ratio];
%
%
%		%	plot(x1, part1);hold on;
%		%	plot(x2, part2,'r');
%		%	plot(x1, [x1, ones(inc + 1,1)]*vec1(:));
%		%	plot(x2, [x2, ones(inc + 1,1)]*vec2(:));
%
%		%	input('');
%
%
%		end
%	end	
%
%	curvature_function = diff(data);
%	curvature_function = curvature_function/max(curvature_function);
%
%	plot(data,'r'); hold on;
%	plot(curvature_function,'k');
%	input('')

	%curvature_function = (curvature_function == max(curvature_function))
	%cutoff_point = find(curvature_function)
	%curvature_function = curvature_function*max(data)/max(curvature_function);



% -------------------   ANOTHER APPROACH 	222

	y = data(:);
	max_data = max(y);
	min_data = min(y);
	inc = (max_data-min_data)/length(y);
	x = [min_data:inc:(max_data - inc)]';
	A = [x ones(length(x),1)];

	[q r] = qr(A);
	coef = r\(q'*y);
	line = A*coef;

	acc = [0;0;diff(diff(y))];
	acc = acc/max(acc);

	vel = [0;diff(y)];
	vel(floor(length(vel)/2):end) = 0;
	vel = vel/max(vel);

	%vel = vel.*acc;

	result = abs(y - line);
	result(floor(length(result)/2):floor(length(result))) = 0;
	result = result/max(result);
	result = result + 3*vel;
	result(floor(length(result)/2):end) = 0;
	result = result/max(result);


	%fir_filter = fir1(10,0.3);
	%result =filter(fir_filter,1,result);

	%peak_map = zeros(size(result));
	cutoff_point = 10;

%	[PKS LOC EXTRA] = findpeaks(result);
%	[LOC, idx]  = sort(LOC);
%	PKS = PKS(idx);
%	if(length(LOC) == 1)
%		cutoff_point = LOC(1);
%		'---------------'
%	elseif((PKS(2) > PKS(1)) & ((LOC(2) - LOC(1)) < 6) | (LOC(1) < 20))
%		cutoff_point = LOC(2);
%		%peak_map(LOC(2)) = 0.8;
%	else
%		cutoff_point = LOC(1);
%		%peak_map(LOC(1)) = 0.8;
%	end
%	
%	if(cutoff_point > 20)
%		cutoff_point = floor(cutoff_point*2/3);
%	end


%	plot(y,'r'); hold on;
%	%plot(line,'b');
%	plot(result,'k');
%	%plot([cutoff_point,cutoff_point],[0,0.8]);
%	axis([0,length(y),0,1.2]);
%	text(length(y) - length(y)/5, 1.1, ['Cutoff : ' num2str(cutoff_point)]);

%	input('');



% -------------------   ANOTHER APPROACH 333



%	inc = 20;
%	current_state = 'initializing';
%
%	for m = 1:length(data)
%	%for m = 60:60
%		if (m-inc) < 1
%			curvature_function = [curvature_function, 0];
%		elseif ((m+inc) > length(data))
%			curvature_function = [curvature_function, 0];
%		else
%			part1 = data(m - inc:m);
%			part2 = data(m:m + inc);
%
%			max_1 = max([part1 part2]);
%			min_1 = min([part1 part2]);
%			x = [min_1:(max_1 - min_1)/inc:max_1]';
%
%
%			%size(part1)
%			%size(x)
%			%plot(1:length(part1), part1,'r'); hold on;
%			%plot(length(part1):length(part1) + length(part2) - 1, part2,'b');
%
%			A = [x, ones(length(x),1)];
%			[q r] = qr(A);
%			vec1 = r\(q'*part1);
%			vec2 = r\(q'*part2);
%
%			%test_x = 1:30;
%			%y1 = vec1(1)*x + vec1(2);
%			%y2 = vec2(1)*x + vec2(2);
%			%plot(x,y1, 'r');
%			%plot(x,y2, 'b');
%
%			slope1 = [abs(vec1(1));1];
%			slope2 = [abs(vec2(1));1];
%
%			angle = 360*acos((slope1'*slope2)/(norm(slope1)*norm(slope2)))/(2*pi);
%			if((angle > 30) & (m > 10))
%				curvature_function = [curvature_function acos((slope1'*slope2)/(norm(slope1)*norm(slope2)))];
%				current_state = 'started';
%			else
%
%				%if(strcmp(current_state,'started'))
%				if(m > 30)
%					break
%				end
%				curvature_function = [curvature_function 0];
%			end
%			%input('')
%		end
%	end
%
%	%curvature_function = (curvature_function == max(curvature_function))
%	%cutoff_point = find(curvature_function)
%	%curvature_function = curvature_function*max(data)/max(curvature_function);



end

