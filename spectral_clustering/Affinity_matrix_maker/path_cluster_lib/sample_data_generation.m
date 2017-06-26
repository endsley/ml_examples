

function [y_normalized, y_total, N, labels, time_series_data] = sample_data_generation(data_set_id, plot_data)
	original_view = 1;
	number_of_data_per_type = 20;
	time_series_data = 1;

	if(plot_data == 1)
		%figure(1, "position", get(0,"screensize")([3,4,3,4]).*[0 0 0.4 0.4]);
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
		labels = [ones(1,number_of_data_per_type+1)];
	
	
		%	Distribution 2
		x1 = [0:7:97]';
		y1 = [-5,0,0,1,2,4,2,4,2,2,0,6,5,4]';
		[y_1,y_normalized_2] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
	
		if plot_data == 1
			plot_noisy(y_normalized_2,'--r',original_view); hold on;
		end
	
		y_total = [y_total, y_1];
		y_normalized = [y_normalized, y_normalized_2];
		labels = [labels 2*ones(1,number_of_data_per_type+1)];
	
	
		%	Distribution 3
		x1 = [0:7:97]';
		y1 = [-5,0,0,1,2,7,7,7,8,8,8,8,10,25]';
		[y_1, y_normalized_3] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
	
		if plot_data == 1
			plot_noisy(y_normalized_3,'--r',original_view); hold on;
		end
	
		y_total = [y_total, y_1];
		y_normalized = [y_normalized, y_normalized_3];
		labels = [labels 3*ones(1,number_of_data_per_type+1)];
	
	
		%	Distribution 4
		x1 = [0:7:97]';
		y1 = [-5,0,0,1,5,5,4,5,6,6,3,3,3,3]';
		[y_1, y_normalized_4] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
	
		if plot_data == 1
			plot_noisy(y_normalized_4,'--r',original_view); hold on;
		end
	
		y_total = [y_total, y_1];
		y_normalized = [y_normalized, y_normalized_4];
		labels = [labels 4*ones(1,number_of_data_per_type+1)];

		save('./data/data_6.mat','y_total', 'y_normalized_1','y_normalized_2','y_normalized_3','y_normalized_4', 'labels');
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
		labels = [ones(1,number_of_data_per_type+1)];
		
		%	Distribution 2
		x1 = [0:4:52]';
		y1 = [-30,0,0,1,2,2,3,4,4,4,5,4,0,-30]';
		[y_2,y_normalized_2] = generate_distribution(number_of_data_per_type, sigma1, sigma2,x1,y1);
		if plot_data == 1
			plot_noisy(y_2,'--g',original_view); hold on;
		end
		labels = [labels 2*ones(1,number_of_data_per_type+1)];
		
		%	Distribution 3
		x1 = [46:3:86]';
		y1 = [-30,0,0,1,2,5,5,7,13,7,5,4,0,-30]';
		[y_3,y_normalized_3] = generate_distribution(number_of_data_per_type, 0.1, 0.5, x1,y1);
		if plot_data == 1
			plot_noisy(y_3,'--c',original_view); hold on;
		end
		labels = [labels 3*ones(1,number_of_data_per_type+1)];
		
		%	Distribution 4
		x1 = [60:3:99]';
		y1 = [-15,0,0,0,2,5,5,7,8,7,5,4,0,-10]';
		[y_4,y_normalized_4] = generate_distribution(number_of_data_per_type, sigma1, sigma2, x1,y1);
		if plot_data == 1
			plot_noisy(y_4,'--b',original_view); hold on;
		end	
		labels = [labels 4*ones(1,number_of_data_per_type+1)];
	
		y_total = [y_1, y_2, y_3, y_4];
		y_normalized = [y_normalized_1, y_normalized_2, y_normalized_3, y_normalized_4];
	elseif(data_set_id == 3)	
		load('./data/data_3.mat');
		y_normalized = y_total;

		number_of_data_per_type = 70;
		labels = [ones(1,number_of_data_per_type+1)];
		labels = [labels 2*ones(1,number_of_data_per_type+1)];
		labels = [labels 3*ones(1,number_of_data_per_type+1)];
		labels = [labels 4*ones(1,number_of_data_per_type+1)];



		
%		%	Distribution 1
%		x1 = [20:3:60]';
%		y1 = [-30,1,2,3,5,6,6,6,4,4,0,-40,-50,-60]';
%		[y_1,y_normalized_1] = generate_distribution(number_of_data_per_type, 0.5, 0.5,x1,y1);
%		labels = [ones(1,number_of_data_per_type+1)];
%		
%		%	Distribution 2
%		x1 = [60:3:99]';
%		%x1 = [17:3:57]';
%		%x1 = [0:4:52]';
%		y1 = [-30,1,2,3,5,6,6,6,4,4,0,-40,-50,-60]';
%		[y_2,y_normalized_2] = generate_distribution(number_of_data_per_type, 0.5, 0.5,x1,y1);
%		labels = [labels 2*ones(1,number_of_data_per_type+1)];
%		
%		%	Distribution 3
%		x1 = [0:4:52]';
%		y1 = [0,0,0,1,2,2,2,2,2,2,2,2,0,0]';
%		[y_3,y_normalized_3] = generate_distribution(number_of_data_per_type, 0.5, 0.5, x1,y1);
%		labels = [labels 3*ones(1,number_of_data_per_type+1)];
%		
%		%	Distribution 4
%		x1 = [60:3:99]';
%		%y1 = [-15,0,0,0,2,5,5,7,8,7,5,4,0,-10]';
%		y1 = [0,0,0,1,2,2,2,2,2,2,2,2,0,0]';
%		[y_4,y_normalized_4] = generate_distribution(number_of_data_per_type, 0.5, 0.5, x1,y1);
%		labels = [labels 4*ones(1,number_of_data_per_type+1)];
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
	elseif(data_set_id == 5)		% gaussian dots 4 clusters
		time_series_data = 0;
		load('./data/data_5.mat');
		y_normalized = y_total;
		each_sample_num = size(y_normalized,2)/4;

		labels = [ones(1,each_sample_num)];
		labels = [labels 2*ones(1,each_sample_num)];
		labels = [labels 3*ones(1,each_sample_num)];
		labels = [labels 4*ones(1,each_sample_num)];


%		each_sample_num = 90;
%
%		%	Distribution 1
%		y_normalized_1 = randn(2,each_sample_num);
%		labels = [ones(1,each_sample_num)];
%
%		%	Distribution 2
%		y_normalized_2 = 2*randn(2,each_sample_num) + 10;
%		labels = [labels 2*ones(1,each_sample_num)];
%
%		%	Distribution 3
%		y_normalized_3 = 8*randn(2,each_sample_num) + 20;
%		labels = [labels 3*ones(1,each_sample_num)];
%
%		%	Distribution 3
%		y_normalized_4 = randn(2,each_sample_num) + 30;
%		labels = [labels 4*ones(1,each_sample_num)];
%
%		%y_total = [p1,p2,p3,p4,q1,q2,q3,r1,r2,r3];
%		y_total = [y_normalized_1,y_normalized_2,y_normalized_3,y_normalized_4];
%		y_normalized = y_total;
%
%		save('./data/data_5.mat','y_total', 'y_normalized_1','y_normalized_2','y_normalized_3','y_normalized_4');
	elseif(data_set_id == 6)
		%load('./data/data_6.mat');
		%y_normalized = y_total;



		time_series_data = 0;
		sample_num = 60;
		%	Distribution 1
		noise = 0.1;
		x = -5:(20/sample_num):5;
		layer_1 = [x, x;sqrt(25 - x.^2) + noise*randn(size(x)), -sqrt(25 - x.^2) + noise*randn(size(x))];
		layer_2 = [sqrt(25 - x.^2) + noise*randn(size(x)), -sqrt(25 - x.^2) + noise*randn(size(x));x, x];

		y_normalized_1 = [layer_1, layer_2];
		labels = [ones(1,size(y_normalized_1,2))];

		%	Distribution 2
		y_normalized_2 = 0.2*randn(2, sample_num) + repmat([2;0],1,sample_num);
		labels = [labels 2*ones(1,sample_num)];

		%	Distribution 3
		y_normalized_3 = 0.2*randn(2, sample_num) + repmat([-2;0],1,sample_num);
		labels = [labels 3*ones(1,sample_num)];

		%	Distribution 4
		y_normalized_4 = 0.2*randn(2, sample_num) + repmat([0;2],1,sample_num);
		labels = [labels 4*ones(1,sample_num)];

		y_total = [y_normalized_1, y_normalized_2, y_normalized_3, y_normalized_4];
		y_normalized = y_total;
		save('./data/data_6.mat','y_total', 'y_normalized_1','y_normalized_2','y_normalized_3','y_normalized_4', 'labels', 'time_series_data');

	elseif(data_set_id == 7)		% 3 clusters
		load('./data/data_7.mat'); 
%		time_series_data = 0;
%
%		%	Distribution 1
%		load('./data/test_data_set.mat'); 
%		y_total = XX{1}';
%		y_normalized = y_total;
%		labels = zeros(1,length(y_total));

	elseif(data_set_id == 8)		% 3 clusters
%		load('./data/data_8.mat'); 
		time_series_data = 0;

		%XX{2} = 3 clusters
		%XX{3} = 3 clusters  - didn't do well
		%XX{4} = 5 clusters
		%XX{5} = 4 clusters
		%XX{6} = 3 clusters

		%	Distribution 1
		load('./data/test_data_set.mat'); 
		y_total = XX{2}';
		y_normalized = y_total;
		labels = zeros(1,length(y_total));

	elseif(data_set_id == 9)		% 3 clusters
%		load('./data/data_9.mat'); 

		time_series_data = 0;
		sample_num = 40;

		%	Distribution 1
		y_normalized_1 = 2*rand(2,sample_num);
		labels = [ones(1,sample_num)];

		%	Distribution 2
		y_normalized_2 = 0.1*rand(2,sample_num)+0.5;
		labels = [labels 2*ones(1,sample_num)];

		%	Distribution 3
		y_normalized_3 = 0.1*rand(2,sample_num)+[0.6;1.3];
		labels = [labels 3*ones(1,sample_num)];

%		y_total = [y_normalized_1, y_normalized_2];
		y_total = [y_normalized_1, y_normalized_2, y_normalized_3];
		y_normalized = y_total;
%		save('./data/data_9.mat','y_total', 'y_normalized_1','y_normalized_2', 'labels', 'y_normalized', 'time_series_data');
		save('./data/data_9.mat','y_total', 'y_normalized_1','y_normalized_2', 'y_normalized_3', 'labels', 'y_normalized', 'time_series_data');
	elseif(data_set_id == 10)		% 3 clusters
%		load('./data/data_8.mat'); 
		time_series_data = 0;
		%	Distribution 1
		load('./data/test_data_set.mat'); 
		y_total = XX{5}';
		y_normalized = y_total;
		%labels = zeros(1,length(y_total));

		load('./data/assignments_10.mat'); 
		labels = assignment;
		save('./data/data_10.mat','y_total', 'y_normalized_1','y_normalized_2', 'y_normalized_3', 'labels', 'y_normalized', 'time_series_data');
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
