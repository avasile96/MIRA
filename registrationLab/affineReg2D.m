function [ Iregistered, M] = affineReg2D( Imoving, Ifixed )
%Example of 2D affine registration
%   Robert Martí  (robert.marti@udg.edu)
%   Based on the files from  D.Kroon University of Twente 

Im=Imoving;
If=Ifixed;

mtype = 'gcc'; % metric type: sd: ssd gcc: gradient correlation; cc: cross-correlation
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
        scale = [1 1 0.1 1 1 0.00001 0.00001];
end;
% Rescaling the Registration Parameters
x=x./scale;

Multiresolution_levels = 6;

% Multiresolution
upper=Multiresolution_levels+1;
fixed_vect_res = cell(upper,1);
move_vect_res = cell(upper,1);
store_fixed=Ifixed;
store_moving=Imoving;
for i=1:1:upper
    fixed_vect_res{i}=store_fixed;
    store_fixed = Multiresolution(store_fixed);
    move_vect_res{i}=store_moving;
    store_moving = Multiresolution(store_moving);
end
    
    
% [x]=fminsearch(@(x)affine_registration_function(x,scale,Im,If,mtype,ttype),x,optimset('Display','iter','MaxIter',1500, 'TolFun', 1.000000e-10,'TolX',1.000000e-10, 'MaxFunEvals', 1000*length(x)));
% Optimization
for k=upper:-1:1

    If=fixed_vect_res{k};
    Im=move_vect_res{k};

    [x]=fminsearch(@(x)affine_registration_function(x,scale,Im,If,mtype,ttype),x,optimset('Display','iter','FunValCheck','on','MaxIter',500, 'TolFun', 1.000000e-30,'TolX',1.000000e-30, 'MaxFunEvals', 2000*length(x),'PlotFcns',@optimplotfval));
    
    switch ttype
        case 'a'
        XY_translation_only = [2 2 1 1 1 1 1]; % Scaling only translations in x and y
        x=x.*XY_translation_only;
        case 'r'
        XY_translation_only = [2 2 1]; 
        x=x.*XY_translation_only;
    end
end

x=x./XY_translation_only;
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
subplot(2,2,1), imshow(If);
title("Fixed Image");
subplot(2,2,2), imshow(Im);
title("Moving Image");
subplot(2,2,3), imshow(Icor);
title("Registered Image");
subplot(2,2,4), imshow(abs(If-Icor));
title("Difference Image");
end

