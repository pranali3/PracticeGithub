no_of_boundary_points = 60;
no_of_outputs=4;
curr_file_count=0;
no_of_dataset=6;
oneFolderName = 'E:\friends\sem8\project_related\dataset\new training set\train this' ;
fldrs = dir( oneFolderName );  % list all sub folders of oneFolderName

%% counting number of files
for ii = 1:6
    if fldrs(ii).name(1) == '.'
       continue; % skip '.' and '..' asuuming all other sub folders do not start with .
    end

    if ~fldrs(ii).isdir
       continue; % skip non subfolders entries
    end
    fls = dir( fullfile( oneFolderName, fldrs(ii).name, '*.jpg' ) ); % list all jpg files in subfolder
    for jj = 1:numel( fls )
        no_of_dataset=no_of_dataset+1;
    end
end
%% initialize
target=zeros(no_of_outputs,no_of_dataset);
neural_input_matrix=zeros(no_of_boundary_points-2,no_of_dataset);

%%
for ii = 1:6
    if fldrs(ii).name(1) == '.'
       continue; % skip '.' and '..' asuuming all other sub folders do not start with .
    end

    if ~fldrs(ii).isdir
       continue; % skip non subfolders entries
    end
    
    fls = dir( fullfile( oneFolderName, fldrs(ii).name, '*.jpg' ) ); % list all jpg files in subfolder
    fldrs(ii).name
    for jj = 1:numel( fls )
 
        imgfile = fullfile( oneFolderName, fldrs(ii).name, fls(jj).name ); % read
        rgbImage= im2bw(imread(imgfile));
        
        %% image
%         hsvImage = rgb2hsv(rgbImage);
% % Extract out the H, S, and V images individually
%         hImage = hsvImage(:,:,1);
%         sImage = hsvImage(:,:,2);
%         vImage = hsvImage(:,:,3);
%     
%        
% 	%This is HSV threshold values for skin color : play with these value for differrent colors
%         hueThresholdLow = 0;
%         hueThresholdHigh = 50;
%         saturationThresholdLow =0.23;
%         saturationThresholdHigh = 0.68;
%         valueThresholdLow = graythresh(vImage);
%         valueThresholdHigh = 1.0;
%     
%     %apply the threshold
%     hueMask = (hImage >= hueThresholdLow) & (hImage <= hueThresholdHigh);
% 	saturationMask = (sImage >= saturationThresholdLow) & (sImage <= saturationThresholdHigh);
% 	valueMask = (vImage >= valueThresholdLow) & (vImage <= valueThresholdHigh);
%     
%     %recombine the HSV
% 	coloredObjectsMask = uint8(hueMask & saturationMask & valueMask);
%     
%     %remove small areas that are noise using bwareaopen
%     smallestAcceptableArea = 1000;  %playaround with this value 
%     coloredObjectMask = bwareaopen(coloredObjectsMask, smallestAcceptableArea);
%     coloredObjectMask = uint8(bwpropfilt(coloredObjectMask,'FilledArea',1));
%     
%     %smooth out the edge using a disky
%     structuringElement = strel('disk', 3);
% 	coloredObjectsMask = imclose(coloredObjectsMask, structuringElement);
% 
%     %fill any holes with white
%     coloredObjectsMask = imfill(logical(coloredObjectsMask), 'holes');
% 	 % e.g. "1.png"
%      
    %% finding slopes
    

    bwimage = rgbImage;
    %imshow(bwimage);
    %boundaries = bwboundaries(bwimage);	

    %For finding the bottom most leftpoint on the boundary
    [white_points_row , white_points_col] = find(bwimage);
    row_index_max = max(white_points_row);
    col_index_min = find(bwimage(row_index_max,:),1);
    start_boundary_point = [row_index_max,col_index_min];

    % extracting selected points from complete boundary
    thisBoundary = bwtraceboundary(bwimage,start_boundary_point,'N',8,Inf,'clockwise');
    %fls(jj).name
    step = floor(length(thisBoundary)/no_of_boundary_points) ;
    feature_points = thisBoundary( step:step:step*no_of_boundary_points , : );
    %feature_points_xy=horzcat(x,y);

    %for finding slopes
    y=feature_points(:,1);
    x=feature_points(:,2);
    feature_points_xy = horzcat(x,y);  
    len = numel(x);
    
    angle_points = ones(len-2,1);
    for i = 1:len - 2
        p0=[x(i);y(i)];
        p1=[x(i+1);y(i+1)];
        p2=[x(i+2);y(i+2)];
        v1=(p1-p0);
        v2=(p2-p1);
        angle_points(i,1) = 180-atan2d((det([v1,v2])),dot(v1,v2));
        line([x(i+1),x(i)] , [y(i+1) ,y(i)] , 'Color' , 'b' , 'LineWidth' , 2 ) ;
    end

    %% creating target and input to neural
    curr_file_count=curr_file_count+1;
    target(ii-2,curr_file_count)=1;
    neural_input_matrix(:,curr_file_count)=angle_points;
    
%     if (ii==3 && jj==1)
%         neural_input_matrix=slope_points;
%     else
%         neural_input_matrix=horzcat(neural_input_matrix,slope_points);
%     end
    
    end
end
%% neural network
%setdemorandstream(491218382);
net=patternnet(10);
[net,tr] = train(net,neural_input_matrix,target);
nntraintool;
genFunction(net,'PraNN');
%testX=angle_points_test;
%testY=net(testX);
%testIndices=vec2ind(testY);
%answer_in_alpha=char(65+testIndices);
   
