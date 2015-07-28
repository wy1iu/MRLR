% ===============================================================================
%   Reference:
%   
%   Misalignment-robust Face Recognition via Efficient Locality-constrained Representation,
%   Yandong Wen, Weiyang Liu, Meng Yang, Yuli Fu, Zhifeng Li
%  
%   Written by Yandong Wen @ SIAT
%   July, 2015
% ===============================================================================

% ===============================================================================
%   Explaination:
%
%   There are Extended Yale B database, which includes 2414 images of 38 subjects 
%   We use the uncropped images of 28 subjects in experiments. 
%   All the training images are pre-aligned with each other using RASL
%
%   1 - Make sure you install Computer Vision System Toolbox in Matlab to use Viola-Jones face detector. 
%   2 - 'Demo' displays the alignment algorithm in MRLR, including MRLR1 and MRLR2.
%   3 - This is not an optimized version.
% ================================================================================
%   Variables:
%              Do :  the pre-aligned dictionary of 28 subjects.
%         imgSize :  the image size of training images
%   transformType :  indicate transformation group, we use 'Similarity' in
%                    our experiment
%        flagMRLR :  flag to select MRLR1 or MRLR2
% ================================================================================
%   For more details, please refers to 'Misalignment-robust Face Recognition
%   via Efficient Locality-constrained Representation'
% ================================================================================

clc;clear; close all;

% % addpath
addpath RASL_toolbox;
addpath utilities;

% % Load the pre-aligned dictionary
load('results\Yale\final.mat');
Do  =  Do * diag(1./sqrt(sum(Do.*Do)));

% % Setting
imgSize = [80, 70];
transformType = 'SIMILARITY';
image_path='test_images/';
img_list=dir([image_path,'*.pgm']);
flagMRLR = 2;  % choose MRLR1 or MRLR2

% % Use viola-jones face detector to obtain initial \tau
faceDetector = vision.CascadeObjectDetector;
for i = 1:size(img_list,1)

    % % Load Image
    test_image =  im2double(imread([image_path,img_list(i).name]));
    bboxes     =  step(faceDetector, test_image);
    
    
    % % Calculate points of eyes with respect to box
    eyePoints  =  [ bboxes(1)+bboxes(3)*0.2581 bboxes(1)+bboxes(3)*0.7319;...
                    bboxes(2)+bboxes(4)*0.3784 bboxes(2)+bboxes(4)*0.3784 ];
    
    % % Align the image by MRLR
    MRLR_main(test_image, imgSize, transformType, Do, eyePoints, flagMRLR);

    pause(3);
    close all;
    
end