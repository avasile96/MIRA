clear all; 
close all;

X=imread('00.png');
%X=rgb2gray(X);
imwrite(X,'piga1.pgm');
Y=imread('01.png');
imwrite(Y,'piga3.pgm');
[num,p_i_primeraImatge,p_i_segonaImatge]=match('piga1.pgm','piga3.pgm');

%% NORANSAC
H=computeHomography(p_i_segonaImatge,p_i_primeraImatge,'affine');
% A = transpose(H);  %Your matrix in here
% t = maketform( 'affine', H);
B = imwarp(Y,H);
C=appendimages(Y,B);

figure;
imshow(B);
figure;
imshow(Y);
figure;
imshow(X);

%% RANSAC
HR=computeHomographyRANSAC(p_i_segonaImatge,p_i_primeraImatge,'affine');
D = transpose(HR);
e= maketform('affine',D);
F=imtransform(Y,e);
G=appendimages(C,F);
figure;
imshow(G);
