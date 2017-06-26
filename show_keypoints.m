%%% Experiments with feature extraction using the Harris corner detector
%%% ****************************************************************
%%% Peter McCloskey
%%% CS 1675 Intro to Computer Vision, University of Pittsburgh 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Need to fix some stuff - the feature descriptors for keypoints are not 
%%% computed and returned by the compute_features function

image_files = dir('*.jpg');      
nfiles = length(image_files);    % Number of files found in MATLAB folder
nfiles = 1;
for i=1:nfiles
    image = imread(image_files(i).name);
    [X, Y, scores, Ix, Iy] = extract_keypoints(image);

end


[X, Y, scores] = compute_features(X, Y, scores, Ix, Iy);
    

function [X, Y, scores, Ix, Iy] = extract_keypoints(image)
%%% Computes the keypoints in an image using Harris feature detection
%%% Arguments:      1. original image
%%%
%%% Returns:        1. X values for keypoints
%%%                 2. Y values for keypoints
%%%                 3. keypoint scores
%%%                 4. X directed gradient values for each pixel
%%%                 5. Y directed gradient values for each pixel
    k = 0.04;
    Threshold = 0.25;

    [dx, dy] = meshgrid(-1:1, -1:1);
    Gxy = fspecial('gaussian',[5,5],1); % gaussian filter


    im = im2double(rgb2gray(image));

    num_rows = size(im, 1);
    num_columns = size(im, 2);

    % Compute x and y derivatives of image
    Ix = conv2(dx, im);
    Iy = conv2(dy, im);


    % Compute products of derivatives at every pixel
    Ix2 = Ix .^ 2;
    Iy2 = Iy .^ 2;
    Ixy = Ix .* Iy;

    % Compute the sums of the products of derivatives at each pixel
    sum_x2 = conv2(Gxy, Ix2);
    sum_y2 = conv2(Gxy, Iy2);
    sum_xy = conv2(Gxy, Ixy);

    R = zeros(num_rows, num_columns);
    
    i = 1;
    for x=1:num_rows,
        for y=1:num_columns,
            % Define at each pixel(x, y) the matrix M
            M = [sum_x2(x, y) sum_xy(x, y); sum_xy(x, y) sum_y2(x, y)];
       
            % Compute the response of the detector at each pixel
            score = det(M) - k * (trace(M) ^ 2);
       
            % Threshold the score
            %if (score > Threshold) && (x ~= 1)  && (y ~= 1)  && (x ~= num_rows)  && (y ~= num_columns)  
            if score > Threshold
                R(x,y) = score;
                scores(i,1) = score;
                Y(i,1) = x; % Set the ith value of the X vector to the current value of x
                X(i,1) = y; % Set the ith value of the Y vector to the current value of y
                i=i+1;
            end
        end
    end
   figure,imshow(image)
   hold on
   plot(X,Y,'ro')
   hold off
end
% features is an nxd matix, each row of which contains the d-dimensional descriptor for the n-th keypoint
function [X, Y, scores] = compute_features(X, Y, scores, Ix, Iy)
%function [features, x, y, scores] = compute_features(x, y, scores, Ix, Iy)
%%% Computes the feature descriptors of keypoints using gradient magnitudes and
%%% angles 
%%% Arguments:      1. X values for keypoints
%%%                 2. Y values for keypoints
%%%                 3. keypoint scores
%%%                 4. X directed gradient values for each pixel
%%%                 5. Y directed gradient values for each pixel
%%%
%%% Returns:        1. X values for keypoints
%%%                 2. Y values for keypoints
%%%                 3. keypoint scores
%%%                 4. keypoint feature descriptors

   % 8 Dimensional Descriptor
    descriptor = zeros(8,size(scores,1));
    features = zeros(size(scores,1),8);
    for i=1:size(Ix,1) % Rows
        for j=1:size(Ix,2) % Columns
            grad_mag = sqrt(Ix(i, j)^2 + Iy(i, j)^2); 
            orient_raw = atand(Iy(i, j) / Ix(i, j)); 
            if(isnan(orient_raw)) 
                assert(grad_mag == 0); 
                orient_raw = 0; % if no change, we won't count a gradient magnitude 
            end
            % Quantize each gradient by incrementing the corresponding
            % orientation bin in the descriptor by the magnitude of the gradient
            if orient_raw >= 0 && orient_raw < 22.5
                descriptor(1) = descriptor(1) + grad_mag;
            elseif orient_raw >= 22.5 && orient_raw < 45
                descriptor(2) = descriptor(2) + grad_mag;
            elseif orient_raw >= 45 && orient_raw < 67.5
                descriptor(3) = descriptor(3) + grad_mag;
            elseif orient_raw >= 67.5 && orient_raw < 90
                descriptor(4) = descriptor(4) + grad_mag;
            elseif orient_raw >= 90 && orient_raw < 112.5
                descriptor(5) = descriptor(5) + grad_mag;
            elseif orient_raw >= 112.5 && orient_raw < 135
                descriptor(6) = descriptor(6) + grad_mag;
            elseif orient_raw >= 135 && orient_raw < 157.5
                descriptor(7) = descriptor(7) + grad_mag;
            elseif orient_raw >= 157.5 && orient_raw < 180
                descriptor(8) = descriptor(8) + grad_mag;
            end
        end
    end
    
    descriptor=descriptor/norm(descriptor,1); % First Normalization
    for i = 1:size(descriptor)
        if descriptor(i) > 0.2
            descriptor(i) = 0.2; % Clip values over 0.2 down to 0.2
        end
    end
    descriptor=descriptor/norm(descriptor,1); % Second Normalization
end


