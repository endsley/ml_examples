

function [y_normalized, y_total, N] = sample_data_generation(data_set_id, plot_data)
	original_view = 1;
	number_of_data_per_type = 20;
	time_series_data = 1;

	if(plot_data == 1)
		figure(1, "position", get(0,"screensize")([3,4,3,4]).*[0 0 0.4 0.4]);
	end

	if(data_set_id == 1)	
		sigma1 = 0.5;
		sigma2 = 0.5;

		%	Distribution 1
		x1 = [0:7:97]';
		y1 = [-5,0,0,1,2,4,2,4,9,9,10,10,20,30]';
		[y_1, y_normalized_1] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
	
		if plot_data == 1
			plot_noisy(y_normalized_1,'--r',original_view); hold on;
		end
	
		y_total = [y_1];
		y_normalized = [y_normalized_1];
	
	
		%	Distribution 2
		x1 = [0:7:97]';
		y1 = [-5,0,0,1,2,4,2,4,2,2,0,6,5,4]';
		[y_1,y_normalized_2] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
	
		if plot_data == 1
			plot_noisy(y_normalized_2,'--r',original_view); hold on;
		end
	
		y_total = [y_total, y_1];
		y_normalized = [y_normalized, y_normalized_2];
	
	
		%	Distribution 3
		x1 = [0:7:97]';
		y1 = [-5,0,0,1,2,7,7,7,8,8,8,8,10,25]';
		[y_1, y_normalized_3] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
	
		if plot_data == 1
			plot_noisy(y_normalized_3,'--r',original_view); hold on;
		end
	
		y_total = [y_total, y_1];
		y_normalized = [y_normalized, y_normalized_3];
	
	
		%	Distribution 4
		x1 = [0:7:97]';
		y1 = [-5,0,0,1,5,5,4,5,6,6,3,3,3,3]';
		[y_1, y_normalized_4] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
	
		if plot_data == 1
			plot_noisy(y_normalized_4,'--r',original_view); hold on;
		end
	
		y_total = [y_total, y_1];
		y_normalized = [y_normalized, y_normalized_4];

	elseif(data_set_id == 2)	
		sigma1 = 0.5;
		sigma2 = 0.5;

		%	Distribution 1
		x1 = [20:3:60]';
		y1 = [-30,0,0,1,2,4,2,4,9,9,5,4,0,-30]';
		[y_1,y_normalized_1] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
		if plot_data == 1
			plot_noisy(y_1,'--r',original_view); hold on;
		end
		
		%	Distribution 2
		x1 = [0:4:52]';
		y1 = [-30,0,0,1,2,2,3,4,4,4,5,4,0,-30]';
		[y_2,y_normalized_2] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
		if plot_data == 1
			plot_noisy(y_2,'--g',original_view); hold on;
		end
		
		%	Distribution 3
		x1 = [46:3:86]';
		y1 = [-30,0,0,1,2,5,5,7,13,7,5,4,0,-30]';
		[y_3,y_normalized_3] = generate_distribution(number_of_data_per_type, 0.1, 0.5, x1,y1);
		if plot_data == 1
			plot_noisy(y_3,'--c',original_view); hold on;
		end
		
		%	Distribution 4
		x1 = [60:3:99]';
		y1 = [-15,0,0,0,2,5,5,7,8,7,5,4,0,-10]';
		[y_4,y_normalized_4] = generate_distribution(number_of_data_per_type, sigma1, sigma2, x1,y1);
		if plot_data == 1
			plot_noisy(y_4,'--b',original_view); hold on;
		end	
	
		y_total = [y_1, y_2, y_3, y_4];
		y_normalized = [y_normalized_1, y_normalized_2, y_normalized_3, y_normalized_4];
	elseif(data_set_id == 3)	
		load('./data/data_3.mat');
		y_normalized = y_total;

		y_normalized(:,1)
		plot(y_normalized(1,:),y_normalized(2,:)); 
		
%		%	Distribution 1
%		x1 = [20:3:60]';
%		y1 = [-30,1,2,3,5,6,6,6,4,4,0,-40,-50,-60]';
%		[y_1,y_normalized_1] = generate_distribution(number_of_data_per_type, 0.5, 0.5,x1,y1);
%		
%		%	Distribution 2
%		x1 = [60:3:99]';
%		%x1 = [17:3:57]';
%		%x1 = [0:4:52]';
%		y1 = [-30,1,2,3,5,6,6,6,4,4,0,-40,-50,-60]';
%		[y_2,y_normalized_2] = generate_distribution(number_of_data_per_type, 0.5, 0.5,x1,y1);
%		
%		%	Distribution 3
%		x1 = [0:4:52]';
%		y1 = [0,0,0,1,2,2,2,2,2,2,2,2,0,0]';
%		[y_3,y_normalized_3] = generate_distribution(number_of_data_per_type, 0.5, 0.5, x1,y1);
%		
%		%	Distribution 4
%		x1 = [60:3:99]';
%		%y1 = [-15,0,0,0,2,5,5,7,8,7,5,4,0,-10]';
%		y1 = [0,0,0,1,2,2,2,2,2,2,2,2,0,0]';
%		[y_4,y_normalized_4] = generate_distribution(number_of_data_per_type, 0.5, 0.5, x1,y1);
%	
%		y_total = [y_1, y_2, y_3, y_4];
%		y_normalized = [y_normalized_1, y_normalized_2, y_normalized_3, y_normalized_4];
%
%		save('./data/data_3.mat','y_total', 'y_normalized_1','y_normalized_2','y_normalized_3','y_normalized_4');
	elseif(data_set_id == 4)
	%	Make sure you run 90% fft filter
	%	use 100% fft
	%	use 0.2 for variance map
		%	Distribution 1
		load('./images/yoga_1a.mat'); p1 = adjust_width(output, 540);
		load('./images/yoga_1b.mat'); p2 = adjust_width(output, 540);
		load('./images/yoga_1c.mat'); p3 = adjust_width(output, 540);
		load('./images/yoga_1d.mat'); p4 = adjust_width(output, 540);

		%	Distribution 2
		load('./images/yoga_2a.mat');  q1 = adjust_width(output, 540);
		load('./images/yoga_2b.mat');  q2 = adjust_width(output, 540);
		load('./images/yoga_2c.mat');  q3 = adjust_width(output, 540);


		%	Distribution 3
		load('./images/yoga_3a.mat');  r1 = adjust_width(output, 540);
		load('./images/yoga_3b.mat');  r2 = adjust_width(output, 540);
		load('./images/yoga_3c.mat');  r3 = adjust_width(output, 540);

		%y_total = [p1,p2,p3,p4,q1,q2,q3,r1,r2,r3];
		y_total = [p1,p2,p3,p4,q1,q2,q3, r1, r2, r3];
		y_normalized = y_total;
	elseif(data_set_id == 5)
		time_series_data = 0;
		each_sample_num = 90;

		%	Distribution 1
		y_normalized_1 = randn(2,each_sample_num);

		%	Distribution 2
		y_normalized_2 = 2*randn(2,each_sample_num) + 10;

		%	Distribution 3
		y_normalized_3 = 8*randn(2,each_sample_num) + 20;

		%	Distribution 3
		y_normalized_4 = randn(2,each_sample_num) + 30;

		%y_total = [p1,p2,p3,p4,q1,q2,q3,r1,r2,r3];
		y_total = [y_normalized_1,y_normalized_2,y_normalized_3,y_normalized_4];
		y_normalized = y_total;

		if plot_data == 1
			plot(y_normalized_1(1,:), y_normalized_1(2,:), 'rx');hold on;
			plot(y_normalized_2(1,:), y_normalized_2(2,:), 'bx');
			plot(y_normalized_3(1,:), y_normalized_3(2,:), 'gx');
			plot(y_normalized_4(1,:), y_normalized_4(2,:), 'cx');hold off;
		end	
	elseif(data_set_id == 6)
		time_series_data = 0;
		sample_num = 50;

		%	Distribution 1
		noise = 0.1;
		x = -5:(20/sample_num):5;
		layer_1 = [x, x;sqrt(25 - x.^2) + noise*randn(size(x)), -sqrt(25 - x.^2) + noise*randn(size(x))];
		layer_2 = [sqrt(25 - x.^2) + noise*randn(size(x)), -sqrt(25 - x.^2) + noise*randn(size(x));x, x];

		y_normalized_1 = [layer_1, layer_2];

		%	Distribution 2
		y_normalized_2 = 0.2*randn(2, sample_num) + repmat([2;0],1,sample_num);

		%	Distribution 3
		y_normalized_3 = 0.2*randn(2, sample_num) + repmat([-2;0],1,sample_num);

		%	Distribution 4
		y_normalized_4 = 0.2*randn(2, sample_num) + repmat([0;2],1,sample_num);

		y_total = [y_normalized_1, y_normalized_2, y_normalized_3, y_normalized_4];
		y_normalized = y_total;
	elseif(data_set_id == 7)
		time_series_data = 0;

		%	Distribution 1
		load('./path_cluster_lib/Data6.mat'); 
		y_total = XX{5}';
		y_normalized = y_total;

		%plot(y_total(1,:), y_total(2,:),'o');
	elseif(data_set_id == 8)
		time_series_data = 1;

		%	Distribution 1
		load('./data/gene_1.mat'); 
		y_total = gene_1';
		y_normalized = y_total;

	elseif(data_set_id == 9)
		time_series_data = 1;

		%	Distribution 1
		load('./data/gene_3.mat'); 
		y_total = gene_3';
		y_normalized = y_total;

	end

	
	if plot_data == 1
		if(time_series_data == 1)
			figure(5, "position", get(0,"screensize")([3,4,3,4]).*[0.4 0.5 0.5 0.6]);
%			plot_noisy(y_normalized_1,'k',0);
%			plot_noisy(y_normalized_2,'k',0);
%			plot_noisy(y_normalized_3,'k',0);
%			plot_noisy(y_normalized_4,'k',0);

			plot_noisy(y_normalized_1,'--r',0);
			plot_noisy(y_normalized_2,'--b',0);
			plot_noisy(y_normalized_3,'--g',0);
			plot_noisy(y_normalized_4,'--c',0);

		else
			figure(5, "position", get(0,"screensize")([3,4,3,4]).*[0.4 0.5 0.5 0.6]);
			plot(y_normalized_1(1,:),y_normalized_1(2,:),'bx',0); hold on;
			plot(y_normalized_2(1,:),y_normalized_2(2,:),'bx',0);
			plot(y_normalized_3(1,:),y_normalized_3(2,:),'bx',0);
			plot(y_normalized_4(1,:),y_normalized_4(2,:),'bx',0); hold off;
		end
	end

	N = size(y_normalized,2);
end
