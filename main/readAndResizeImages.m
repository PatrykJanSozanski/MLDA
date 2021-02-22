function img = readAndResizeImages(filename)
%
% Reads and resizes images.
%
% Inputs:
% > filename                - image to be read and resized
%   
% Outputs:
% > img                     - resized image

% read image
im = imread(filename);

% resize image
img = imresize(im,[300 300]);
end