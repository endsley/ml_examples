#!/usr/bin/octave --silent




addpath('./path_cluster_lib')
	
%	initialization
	%row_v = 17;
	cluster_N = 4;
	model_order = 8;
	sample_data_id = 10;

	[y_normalized, y_total, N, labels, time_series_data] = sample_data_generation(sample_data_id, 0);
	Euclid_matrix = calc_Eucli_Distance_matrix(y_normalized, 0);
	data{1} = y_normalized;

	fir_filter = fir1(19,0.20);
	%[b,a]=butter ( 20, 0.5 );
 
	cluster_matrix = form_cluster_matrix(labels);
	adjacency_matrix = [];
	
	for row_v = 1:size(y_normalized,2)
	%for row_v = 4:4
		row_v
		data_point = y_total(:, row_v);
		row1 = Euclid_matrix(row_v,:);
		cluster_id = cluster_matrix(row_v, :);

		[s, idx] = sort(row1);
		new_s = s./max(s);

%		new_s = filter(b,a,new_s);
%		new_s = filter(new_s, fir_filter);
		new_s = conv(new_s,fir_filter,'same');
%		new_s(end-10:end) = new_s(end-10);


		%best_fit = detrend(new_s,1);
		%best_fit = abs(fft(best_fit));
		%best_fit = best_fit/max(best_fit);
		%best_fit(1:10)

		%v = [1:length(new_s)]';
		%A = [ones(length(new_s), 1), v v.^2 v.^3 v.^4 v.^5 v.^6 v.^7 v.^8 v.^9 v.^10 v.^11 v.^12 v.^13];
		%A = [ones(length(new_s), 1), v v.^2 v.^3 v.^4 v.^5 v.^6 v.^7 v.^8];
		%[q r] = qr(A);
		%coef = r\(q'*new_s');
		%best_fit = A*coef;

		%lambda = 8.0;      % regularization parameter
		%rel_tol = 0.00001;     % relative target duality gap
		%[coef,status]=l1_ls(A,new_s',lambda,rel_tol);
		%best_fit = A*coef;


		%[coef, S] = polyfit([1:length(new_s)], new_s, 6)
		%best_fit = S.X*coef';
		%best_fit = best_fit/max(best_fit);

			%	use predefined cut off
		cluster_id = cluster_id(idx);
		mm = cumsum(cluster_id);
		cutoff = find(diff(mm == max(mm)))

			%	use first cutoff
		%curvature_function = get_menger_curvature(new_s);
		%[cutoff, filtered_curv] = get_peaks(curvature_function, new_s);
		%cutoff = ceil(cutoff*0.8);
		%%cutoff = 14;
		cutoff_point = s(cutoff);
		
		new_row = row1 < cutoff_point;
		adjacency_matrix = [adjacency_matrix; new_row];

		plot(new_s,'r'); hold on;
		%plot(new_row/3 + 1/3,'rx');
		plot([cutoff,cutoff],[0,0.8]);
		text(length(new_s) - length(new_s)/2, 0.9, ['Cutoff : ' num2str(cutoff) ' , ' num2str(cutoff_point)]);
		plot(cluster_id,'o');
		%plot(best_fit,'g')
		%plot(filtered_curv, 'g');
		if(length(data_point) == 2)
			text(length(new_s) - length(new_s)/2, 0.8, ['Point : ' num2str(data_point(1)) ' , ' num2str(data_point(2))]);
		end
		%plot(curvature_function,'k');
		print(['./distance_path/Figure_' num2str(row_v) '.png'], '-S600,400');
		hold off;
		%input('');
		

%		fh = figure(3);
%		%set(fh, 'visible','Off')
%		plot(s,'r'); hold on;
%		%plot(10*curvature_function);
%		%plot(10*abs(diff(row1)));
%		%D = [0 0 abs(diff(diff(row1)))] + [0 abs(diff(row1))] + curvature_function;
%		D =  max(s)*curvature_function./max(curvature_function);
%		D(1:10) = 0;
%		plot(D);
%		print(['./distance_path/Figure_' num2str(row_v) '.png'])
%		hold off;
%		input('')
	end





	%load('data/assignments_7.mat');
	%time_series_data = 0;
	%sample_data_id = 7;
	%cluster_N =3;



	Mx = adjacency_matrix.*adjacency_matrix';
	[centroid, pointsInCluster, assignment] = spectral_fit(Mx, cluster_N);

%	%D = diag(sum(Mx).^(-1/2));
%	%Laplacian = D*Mx*D;
%	Adjacency_matrix = Mx - diag(diag(Mx));

	%assignment = 4*ones(size(data{1},2),1);
	%[clusts,best_group_index,Quality,Vr] = cluster_rotate(Mx,[2:model_order],1,1);
	%NN = size(clusts{best_group_index},2);
	%for m = 1:NN
	%	p = clusts{best_group_index};
	%	assignment(p{1,m}) = m;
	%end

	save(['./data/assignments_' num2str(sample_data_id) '.mat'],'assignment', 'adjacency_matrix','sample_data_id' );

	x = [1:size(data{1},1)]';
	if(time_series_data)
		plot_cluster_results(x, assignment, data{1}, N, 10);
	else
		plot_2d_data(assignment, data{1}, N, 10);
	end


		input('')


%
%	y = c1(idx)
%
%	conv = abs(conv(y,[1,-1],'same'));
%	if( sum(conv) == 1 )
%		toggle_indx = find(conv);
%		y(toggle_indx + 1) = 0;
%		y(toggle_indx + 2) = 1;
%		y(toggle_indx + 3) = 0;
%		y(toggle_indx + 4) = 1;
%		y(toggle_indx + 5) = 0;
%	end
%
%
%	x = [1:length(y)]';
%	
%	[theta, beta] = logistic_regression(y,x,0);
%	v = theta - beta*x;
%	out = exp(-v)./(1+exp(-v));
%	plot(y,'o'); hold on;
%	plot(out,'r');
%	plot(s);
%	input('');








%%	plotting
%	x = [1:size(y_normalized,1)]';
%	plot_cluster_results(x, assignment, data{1}, N, 10);
%	plot_cluster_results(x, assignment, original{1}, N, 11);
%	%plot_2d_data(assignment, original{1}, N, 10);
%
%	input('');

