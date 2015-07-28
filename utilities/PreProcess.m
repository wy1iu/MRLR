function proceImg = PreProcess(img, transformation, varargin )
% ===============================================================================
%   Reference:
%   
%   Misalignment-robust Face Recognition via Efficient Locality-constrained Representation,
%   Yandong Wen, Weiyang Liu, Meng Yang, Yuli Fu, Zhifeng Li
%  
%   Written by Yandong Wen @ SCUT
%   July, 2015
% ===============================================================================

if isempty(varargin)
    numScales = 1;
	gammaType = 'linear';
	sigma0 = 2/5;
    sigmaS = 10;
elseif length(varargin)==1
    numScales = varargin{1};
    gammaType = 'linear';
	sigma0 = 2/5;
    sigmaS = 10;
elseif length(varargin)==2
    numScales = varargin{1};
    gammaType = varargin{2};
    sigma0 = 2/5;
    sigmaS = 10;
elseif length(varargin)==3
    numScales = varargin{1};
    gammaType = varargin{2};
    sigma0 = varargin{3};
    sigmaS = 10;
elseif length(varargin)==4
    numScales = varargin{1};
    gammaType = varargin{2};
    sigma0 = varargin{3};
    sigmaS = varargin{4};
end

currentImage = gamma_decompress(img, gammaType);
% proceImgPyramid = gauss_pyramid( currentImage, numScales,...
%         sigma0, sigmaS );
proceImgPyramid = gauss_pyramid( currentImage, numScales,...
        sqrt(det(transformation(1:2,1:2)))*sigma0, sigmaS );
proceImg = proceImgPyramid{1};

end

