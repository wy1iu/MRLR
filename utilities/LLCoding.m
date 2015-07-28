function tau = LLCoding( D, J, invT1, y )
% ===============================================================================
%   Reference:
%   
%   Misalignment-robust Face Recognition via Efficient Locality-constrained Representation,
%   Yandong Wen, Weiyang Liu, Meng Yang, Yuli Fu, Zhifeng Li
%  
%   Written by Yandong Wen @ SIAT
%   July, 2015
% ===============================================================================
    T3 = J'*J;
    T2 = D'*J;
    tem_T = T2'*invT1;
    Z2 = T3 - tem_T*T2; 
    tau = Z2\(tem_T*(D'*y)-J'*y);    

end

