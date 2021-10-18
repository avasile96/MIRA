% clean
clear all; close all; clc;

tic;

% Read two imges 
Imoving=im2double(rgb2gray(imread('brain4.png'))); 
Ifixed=im2double(rgb2gray(imread('brain3.png')));

Im=Imoving;
If=Ifixed;

[ Iregistered, M] = affineReg2D( Imoving, Ifixed );

toc;
