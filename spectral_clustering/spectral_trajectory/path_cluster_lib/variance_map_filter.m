
%	each column of y is a single dataset, this function will remove
%	the least import parts of the data
function filter_out = variance_map_filter(y, remove_percentage)
	if(remove_percentage == 0)
		filter_out = y;
		return;
	end

	vMap = std(y');
	vMap = vMap/sum(vMap);
	[s,id] = sort(vMap,'descend');

	cdf = cumsum(s);
	filter_out = y(id(cdf > remove_percentage),:);
	len = size(filter_out,1);

	filter_out = filter_out./repmat(sum(filter_out), len,1);


end
