function out= Multiresolution(in)

    Image_Size = size(in);
    out = [];
    
    Filtering = imgaussfilt(in,2); % Necessary for information preservation
    out(:,:) = Filtering(1:2:Image_Size(1),1:2:Image_Size(2)); % Subsampling stage

end