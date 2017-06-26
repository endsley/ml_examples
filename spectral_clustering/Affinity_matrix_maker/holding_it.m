function curvature_function = get_chieh_curvature(data)
	data = data(:);
	inc = 20;
	curvature_function = [];
	current_state = 'initializing';

	for m = 1:length(data)
	%for m = 60:60
		if (m-inc) < 1
			curvature_function = [curvature_function, 0];
		elseif ((m+inc) > length(data))
			curvature_function = [curvature_function, 0];
		else
			part1 = data(m - inc:m);
			part2 = data(m:m + inc);

			max_1 = max([part1 part2]);
			min_1 = min([part1 part2]);
			x = [min_1:(max_1 - min_1)/inc:max_1]';


			%size(part1)
			%size(x)
			%plot(1:length(part1), part1,'r'); hold on;
			%plot(length(part1):length(part1) + length(part2) - 1, part2,'b');

			A = [x, ones(length(x),1)];
			[q r] = qr(A);
			vec1 = r\(q'*part1);
			vec2 = r\(q'*part2);

			%test_x = 1:30;
			%y1 = vec1(1)*x + vec1(2);
			%y2 = vec2(1)*x + vec2(2);
			%plot(x,y1, 'r');
			%plot(x,y2, 'b');

			slope1 = [abs(vec1(1));1];
			slope2 = [abs(vec2(1));1];

			angle = 360*acos((slope1'*slope2)/(norm(slope1)*norm(slope2)))/(2*pi);
			if((angle > 30) & (m > 10))
				curvature_function = [curvature_function acos((slope1'*slope2)/(norm(slope1)*norm(slope2)))];
				current_state = 'started';
			else

				%if(strcmp(current_state,'started'))
				if(m > 30)
					break
				end
				curvature_function = [curvature_function 0];
			end
			%input('')
		end
	end

	%curvature_function = (curvature_function == max(curvature_function))
	%cutoff_point = find(curvature_function)
	%curvature_function = curvature_function*max(data)/max(curvature_function);
end
