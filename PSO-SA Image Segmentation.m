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
clusteres = 5;

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
Itr=30;
% SA solver + PSO body
SA_opts = optimoptions('simulannealbnd','display','iter','MaxTime',Itr);
% PSO-SA Run
[centers, Error] = particleswarm(CostFunction, nVar,VarMin,VarMax,SA_opts);

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

%% Plot PSO-SA Segmented Result
disp(['Error Is: ' num2str(Error)]);
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
subimage(MainOrg);title('Original');
subplot(2,2,2)
subimage(Gray);title('Gray');
subplot(2,2,3)
imshow(SA_Segmented,[]);
title(['PSO-SA Gray Segmented, Clusters = ' num2str(clusteres)]);
subplot(2,2,4)
imshow(ColorSeg,[]);
title(['PSO-SA Color Segmented, Clusters = ' num2str(clusteres)]);

% That's it, GoodBye :|
