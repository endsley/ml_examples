
%	kl( v1 || v2 ) = sum v1 ln(v1/v2)
function divergence = kl_divergence(v1,v2,normalizeV)

	if(normalizeV == 1)
		v1 = v1/sum(v1);
		v2 = v2/sum(v2);
	end

	divergence = sum(v1.*log(v1./v2));
	%divergence = sum(v1.*log(v1./v2));
end

