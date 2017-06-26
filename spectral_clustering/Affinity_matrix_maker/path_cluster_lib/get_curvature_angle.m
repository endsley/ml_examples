
function angle = get_curvature_angle(point, curv)

	left_p = point - 5;
	right_p = point - 4;
	y = [curv(left_p); curv(right_p)];

	A = [[left_p;right_p], [1;1]];
	[q r] = qr(A);
	coef = r\(q'*y);
	direction_1 = [1, coef(1)];

	left_p = point - 10;
	right_p = point + 10;
	pp = [[left_p:right_p]', ones(length([left_p:right_p]),1)]*coef;
	%plot( left_p:right_p , pp,'b'); hold on;

	%--------------------

	left_p = point + 4;
	right_p = point + 5;
	y = [curv(left_p); curv(right_p)];

	A = [[left_p;right_p], [1;1]];
	[q r] = qr(A);
	coef = r\(q'*y);
	direction_2 = [1, coef(1)];

	left_p = point - 10;
	right_p = point + 10;
	pp = [[left_p:right_p]', ones(length([left_p:right_p]),1)]*coef;
	%plot( left_p:right_p , pp,'b'); hold on;


	angle = 360*acos((direction_1*direction_2')/(norm(direction_1)*norm(direction_2)))/(2*pi);
end
