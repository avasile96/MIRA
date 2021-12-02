% clean
clear all; close all; clc;

% Read two imges 
Imoving=im2double(rgb2gray(imread('brain1.png'))); 
Ifixed=im2double(rgb2gray(imread('brain2.png')));

Im=Imoving;
If=Ifixed;

[ Iregistered, M] = affineReg2D( Imoving, Ifixed );
