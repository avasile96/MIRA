function [ Iregistered, M] = affineReg2D( Imoving, Ifixed )
%Example of 2D affine registration
%   Robert Martí  (robert.marti@udg.edu)
%   Based on the files from  D.Kroon University of Twente 

Im=Imoving;
If=Ifixed;

mtype = 'cc'; % metric type: sd: ssd gcc: gradient correlation; cc: cross-correlation
ttype = 'a'; % rigid registration, options: r: rigid, a: affine

% Parameter scaling of the Translation and Rotation
% and initial parameters
switch ttype
    case 'r'
        x=[0 0 0]; %vector of x y z movements
        scale = [1 1 0.1]; 
    case 'a'
        x=[0 0 0 1 1 0 0];
      % x =[ translationX translationY, rotate, resizeX, resizeY, shearingXY, searingYX]
        scale = [1 1 0.1 1 1 0.0001 0.0001];
end
% Rescaling the Registration Parameters
x=x./scale;

Multiresolution_levels = 6;

% Multiresolution
Total_Image=Multiresolution_levels+1;
I_ff=Ifixed;
I_mv=Imoving;
my_im_cell_1 = cell(Total_Image,1);
my_im_cell_2 = cell(Total_Image,1);
for i=1:1:Total_Image
    my_im_cell_1{i}=I_ff;
    I_ff = Multiresolution(I_ff);
    my_im_cell_2{i}=I_mv;
    I_mv = Multiresolution(I_mv);
end
    
    
% [x]=fminsearch(@(x)affine_registration_function(x,scale,Im,If,mtype,ttype),x,optimset('Display','iter','MaxIter',1500, 'TolFun', 1.000000e-10,'TolX',1.000000e-10, 'MaxFunEvals', 1000*length(x)));
% Optimization
for k=Total_Image:-1:1
Im=my_im_cell_2{k};
If=my_im_cell_1{k};
[x]=fminsearch(@(x)affine_registration_function(x,scale,Im,If,mtype,ttype),x,optimset('Display','iter','FunValCheck','on','MaxIter',500, 'TolFun', 1.000000e-30,'TolX',1.000000e-30, 'MaxFunEvals', 2000*length(x),'PlotFcns',@optimplotfval));
switch ttype
    case 'a'
    scaleTxTy = [2 2 1 1 1 1 1]; % Only doubling the Tx and Ty, others kept constant.
    x=x.*scaleTxTy;
    case 'r'
    scaleTxTy = [2 2 1]; % Only doubling the Tx and Ty, others kept constant.
    x=x.*scaleTxTy;
end
end
x=x./scaleTxTy;
x=x.*scale;

switch ttype
	case 'r'
        % Make the affine transformation matrix
         M=[ cos(x(3)) sin(x(3)) x(1);
            -sin(x(3)) cos(x(3)) x(2);
           0 0 1];
       
    case 'a'
        % Make the affine transformation matrix
        M_Scalled=[ x(4)*cos(x(3)) sin(x(3)) x(1);
                        -sin(x(3)) x(5)*cos(x(3)) x(2);
                            0 0 1];

        M_shearing=[1 x(6) 0;
                    x(7) 1 0;
                       0 0 1];

        M=M_Scalled*M_shearing;
end
     

 % Transform the image 
Icor=affine_transform_2d_double(double(Im),double(M),0); % 3 stands for cubic interpolation

Iregistered = Icor;

Best_metric=Gradient_cc(Ifixed,Icor);

%Parameters Findings 
Translation_X=x(1);
Translation_Y=x(2);
Rotation=x(3);
disp(Translation_X)
disp(Translation_Y)
disp(Rotation)
fprintf('Best quantative metric (%s) is %.4f \n \n',mtype,Best_metric);

% Show the registration results
figure,
subplot(2,2,1), imshow(If), title('Image fixed');
subplot(2,2,2), imshow(Im), title('Image moving');
subplot(2,2,3), imshow(Icor), title('Image registered');
subplot(2,2,4), imshow(abs(If-Icor)), title('Difference between Image fixed and Image registered');
end

