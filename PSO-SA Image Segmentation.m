%% PSO + SA Image Segmentation
% Empowering traditional clustering techniques with evolutionary
% algorithms, here two strong ones, namely particle swarm optimization and
% simulated annealing are used.
% Enjoy!!!

%% Cleaning the Stage
clc;
clear;
close all;
warning('off');

%% Reading Image
MainOrg=imread('tst.jpg');
Gray=rgb2gray(MainOrg);
InpMat= double(MainOrg);

%% Basics
[s1,s2,s3]=size(InpMat);
R = InpMat(:,:,1);
G = InpMat(:,:,2);
B = InpMat(:,:,3);
X1 = (R-min(R(:)))/(max(R(:))-min(R(:)));
X2 = (G-min(G(:)))/(max(G(:))-min(G(:)));
X3 = (B-min(B(:)))/(max(B(:))-min(B(:)));
X = [X1(:) X2(:) X3(:)];

%% Cluster Numbers
clusteres = 7;

%% Cost Function and Parameters
% Cost Function
CostFunction=@(m) CLuCosPSOSA(m, X, clusteres);  
% Decision Variables
VarSize=[clusteres size(X,2)];  
% Number of Decision Variables
nVar=prod(VarSize);
% Lower Bound of Variables
VarMin= repmat(min(X),1,clusteres);      
% Upper Bound of Variables
VarMax= repmat(max(X),1,clusteres);     

%% PSO-SA Clustering Option and Run
% PSO-SA Options
% Iterations (more value means: slower runtime but, better result)
Itr=50;
% SA solver + PSO body
SA_opts = optimoptions('simulannealbnd','display','iter','MaxTime',Itr,'PlotFcn',@pswplotbestf);
options.SwarmSize = 250;
% PSO-SA Run
disp(['SA-PSO Segmentation Is Started ... ']);
[centers, Error] = particleswarm(CostFunction, nVar,VarMin,VarMax,SA_opts);
disp(['SA-PSO Segmentation Is Ended. ']);

%% Calculate Distance Matrix
% Create the Cluster Center 
g=reshape(centers,3,clusteres)'; 
% Create a Distance Matrix
d = pdist2(X, g); 
% Assign Clusters and Find Closest Distances
[dmin, ind] = min(d, [], 2);
% Sum of Cluster Distance
WCD = sum(dmin);
% Fitness Function of Centers Sum
z=WCD; 
% Final Segmented Image
SA_Segmented=reshape(ind,s1,s2);
PSOSAuint=uint8(SA_Segmented);
ColorSeg = labeloverlay(Gray,PSOSAuint);
%
medgray = medfilt2(SA_Segmented,[5 5]);
%
redChannel = ColorSeg(:,:,1); % Red channel
greenChannel = ColorSeg(:,:,2); % Green channel
blueChannel = ColorSeg(:,:,3); % Blue channel
medcolor1 = medfilt2(redChannel,[4 6]);
medcolor2 = medfilt2(greenChannel,[4 6]);
medcolor3 = medfilt2(blueChannel,[4 6]);
medrgb = cat(3, medcolor1, medcolor2, medcolor3);

%% Plot PSO-SA Segmented Result
disp(['Error Is: ' num2str(Error)]);
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
subimage(MainOrg);title('Original');
subplot(2,3,2)
subimage(Gray);title('Gray');
subplot(2,3,3)
imshow(SA_Segmented,[]);
title(['PSO-SA Gray Segmented, Clusters = ' num2str(clusteres)]);
subplot(2,3,4)
imshow(ColorSeg,[]);
title(['PSO-SA Color Segmented, Clusters = ' num2str(clusteres)]);
subplot(2,3,5)
imshow(medgray,[]);
title(['PSO-SA Gray Median Filtered ']);
subplot(2,3,6)
imshow(medrgb,[]);
title(['PSO-SA Color Median Filtered']);
% That's it, GoodBye :|
