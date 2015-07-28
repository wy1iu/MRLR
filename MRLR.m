function [y, transformation] = MRLR(I0, I0x, I0y, D, c, imgSize,...
                                            transformation, transformType,...
                                            frmCoords, innerIter, display)
% ===============================================================================
%   Reference:
%   
%   Misalignment-robust Face Recognition via Efficient Locality-constrained Representation,
%   Yandong Wen, Weiyang Liu, Meng Yang, Yuli Fu, Zhifeng Li
%  
%   Written by Yandong Wen @ SIAT
%   July, 2015
% ===============================================================================
    tol       = 1e-4;
    maxIter   = innerIter;
    converged = false;
    iter      = 0;
    invT1     = inv(D'*D + diag(c.*c));

    while ~converged

        iter =  iter + 1;

        Tfm  =  fliptform(maketform('projective',transformation'));
        I    =  vec(imtransform(I0, Tfm, 'XData', [1 imgSize(2)], ...
                                         'YData', [1 imgSize(1)],'Size',imgSize));
        Iu   =  vec(imtransform(I0x, Tfm, 'XData', [1 imgSize(2)], ...
                                          'YData', [1 imgSize(1)],'Size',imgSize));
        Iv   =  vec(imtransform(I0y, Tfm, 'XData', [1 imgSize(2)], ...
                                          'YData', [1 imgSize(1)],'Size',imgSize));

        y_w  =  I; %vec(I);
        Iu   =  (1/norm(y_w))*Iu - ( (y_w'*Iu)/(norm(y_w))^3 )*y_w ;
        Iv   =  (1/norm(y_w))*Iv - ( (y_w'*Iv)/(norm(y_w))^3 )*y_w ;
        y_w  =  y_w / norm(y_w) ; % normalize
        tau  =  projective_matrix_to_parameters(transformType,transformation) ; 

        % Compute Jacobian
        J    =  image_Jaco(Iu, Iv, imgSize, transformType, tau);

        % Update del_tau
        del_tau = LLCoding( D, J, invT1, y_w );
        tau     =  tau + del_tau;
        transformation =  parameters_to_projective_matrix( transformType, tau );

        % display
        if display==1
            Tfm = fliptform(maketform('projective', inv(transformation')));
            frm = tformfwd(point2frm(frmCoords(1:2,:))', Tfm )';
            plot(frm(1,:), frm(2,:), 'g-', 'LineWidth', 0.8);
            pause(0.05)
        end

        % %
        if norm(del_tau) <= tol
            converged = true ;
        end
        if ~converged && iter >= maxIter
            converged = true ;       
        end
    end

    Tfm =  fliptform(maketform('projective',transformation'));
    y   =  imtransform(I0, Tfm, 'XData', [1 imgSize(2)], ...
                                'YData', [1 imgSize(1)],'Size',imgSize);
end


