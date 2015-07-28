function frm = point2frm( point )
% ===============================================================================
%   Reference:
%   
%   Misalignment-robust Face Recognition via Efficient Locality-constrained Representation,
%   Yandong Wen, Weiyang Liu, Meng Yang, Yuli Fu, Zhifeng Li
%  
%   Written by Yandong Wen @ SCUT
%   July, 2015
% ===============================================================================
point = point(1:2,:);
frm     = [point, [point(1,1)+point(1,3)-point(1,2);...
                   point(2,1)+point(2,3)-point(2,2)],   point(:,1)];

end

