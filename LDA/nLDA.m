#!/usr/bin/octave

function metrics = nLDA()
	x = -8:8;
	
	n0 = 50;
	n1 = 50;
	
	Dat0 = randn(n0,2);
	Dat1 = randn(n1,2) + 2;
	
	%set_0 = [Dat0, Dat0.^2]
	%set_1 = [Dat1, Dat1.^2]
	
	total_data = [Dat0; Dat1];
	center_bias = mean(total_data);
	Dat0 = Dat0 - repmat(center_bias,n0,1); % + [0 10];
	Dat1 = Dat1 - repmat(center_bias,n1,1); % + [0 10];
	
	set_0 = [Dat0 Dat0.^2  Dat0.^3 Dat0.^4];
	set_1 = [Dat1 Dat1.^2  Dat1.^3 Dat1.^4];
	
	%set_0 = [Dat0 Dat0.^2  Dat0.^3];
	%set_1 = [Dat1 Dat1.^2  Dat1.^3];
	
	%set_0 = [Dat0 Dat0.^2];
	%set_1 = [Dat1 Dat1.^2];
	
	%set_0 = [Dat0];
	%set_1 = [Dat1];
	
	m0 = mean(set_0);
	m1 = mean(set_1);
	
	M0 = set_0 - repmat(m0, n0,1);
	S0 = (M0'*M0)/n0;
	
	M1 = (set_1 - repmat(m1, n1, 1));
	S1 = (M1'*M1)/n1;
	
	S = S1 + S0;
	S_inv = inv(S);
	
	w = S_inv*(m0' - m1');
	w_direction = w/norm(w);
	
	
	project_0 = [set_0*w_direction, zeros(n0,1) ];
	project_1 = [set_1*w_direction, zeros(n1,1) ];
	
	mean_1 = mean(project_0);
	mean_2 = mean(project_1);
	
	var_1 = var(project_0);
	var_2 = var(project_1);
	
	diff_pos = mean_2(1) - mean_1(1);
	metrics = [var_1(1), var_2(1), abs(diff_pos)];
	
	
	%project_0 = kron((set_0*w_direction)',w_direction);
	%project_1 = kron((set_1*w_direction)',w_direction);
	
	
	%plot(set_0(:,1), set_0(:,2), 'bo')
	%hold on;
	%plot(set_1(:,1), set_1(:,2), 'ro')
	
	%plot(x,y)
	%plot(project_0(:,1), project_0(:,2),'go')
	%plot(project_1(:,1), project_1(:,2),'yo')
	
	%plot(project_0(1,:), project_0(2,:),'go')
	%plot(project_1(1,:), project_1(2,:),'yo')
	
	%input('press any key')
end
