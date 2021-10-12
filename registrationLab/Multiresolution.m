function out= Multiresolution(in)

%     Center = 0.6; % 0.6 in the Paper
%     G_distribution = [0.25-Center/2 0.25 Center 0.25 0.25-Center/2]; %Logic was from orginal Paper
%     %Burt and Adelson, "The Laplacian Pyramid as a Compact Image Code," IEEE Transactions on 
%     %Communications, Vol. COM-31, no. 4, April 1983, pp. 532-540.
%     Gaussian_kernel = kron(G_distribution,G_distribution');
%     Image_Size = size(in);
%     out = [];
%     
%     for channel = 1:size(in,3)
% 	    Image_each_Channnel = in(:,:,channel);
% 	    Filtering = imfilter(Image_each_Channnel,Gaussian_kernel,'replicate','same');
% 	    out(:,:,channel) = Filtering(1:2:Image_Size(1),1:2:Image_Size(2));
%     end

    Image_Size = size(in);
    out = [];
    
    Filtering = imgaussfilt(in,2);
    out(:,:) = Filtering(1:2:Image_Size(1),1:2:Image_Size(2));

end