clear; close all; clc

pics = 2;  % To choose which set of images are going to be used
plot_keypoints = 0;  % Set to 1 to show figures with the key points of both images

if pics == 1
    img1 = 'retina1.png';
    img2 = 'retina2.png';
elseif pics == 2
    img1 = 'skin1.jpg';
    img2 = 'skin2.jpg';
end    

I = single(rgb2gray(imread(img1)));
I2 = single(rgb2gray(imread(img2)));

for i = 1.5:3:10.5
    
    [f,d] = vl_sift(I);
    [f2,d2] = vl_sift(I2);
    
    [matches, scores] = vl_ubcmatch(d, d2, i);

    figure; imshow(cat(2, mat2gray(I), mat2gray(I2)));
    title(['Threshold for the VLFeat match function is: ',num2str(i)])

    [drop, perm] = sort(scores, 'descend');
    matches = matches(:, perm);
    scores  = scores(perm);

    xa = f(1,matches(1,:));
    xb = f2(1,matches(2,:)) + size(I,2); 
    ya = f(2,matches(1,:)); 
    yb = f2(2,matches(2,:));

    hold on;
    h = line([xa ; xb], [ya ; yb]);
    set(h,'linewidth', 1, 'color', 'r');

    vl_plotframe(f(:,matches(1,:))); 
    f2(1,:) = f2(1,:) + size(I,2); 
    vl_plotframe(f2(:,matches(2,:))); 
    axis image off;
end

if plot_keypoints
    figure; imshow(I,[])
    h1 = vl_plotframe(f);
    set(h1,'color','y','linewidth',2);
    figure; imshow(I2,[])
    h2 = vl_plotframe(f2);
    set(h2,'color','y','linewidth',2);
end
