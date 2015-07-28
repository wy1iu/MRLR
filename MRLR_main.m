function MRLR_main( test_image, imgSize, transformType, D, eyesPts, flagMRLR )
% ===============================================================================
%   Reference:
%   
%   Misalignment-robust Face Recognition via Efficient Locality-constrained Representation,
%   Yandong Wen, Weiyang Liu, Meng Yang, Yuli Fu, Zhifeng Li
%  
%   Written by Yandong Wen @ SIAT
%   July, 2015
% ===============================================================================

% %  Initialization
    display = 1; 
    sigma = 0.1;
    
    eyeCoords = [ 16.67  53.33  ;
                  28.67  28.67 ];
    frmCoords = [ 1, 70, 70 ;
                  1, 1,  80 ;
                  1, 1,  1 ];
    
    transformation = [ TwoPointSimilarity( eyeCoords, eyesPts );...
                       0 0 1];
    if display == 1
        frm = transformation*frmCoords;
        frm = point2frm(frm(1:2,:));
        imshow(test_image); hold on;
        plot(frm(1,:),frm(2,:),'r-','LineWidth', 2);
    end
    
    I0 = PreProcess(test_image, transformation);
    I0x = imfilter( I0, (-fspecial('sobel')') / 8 );
    I0y = imfilter( I0,  -fspecial('sobel')   / 8 );

    Tfm = fliptform(maketform('projective',transformation'));
    y  = imtransform(I0, Tfm, 'XData', [1 imgSize(2)], ...
                                    'YData', [1 imgSize(1)],'Size',imgSize);
                               
    innerIter = 30;
    outterIter = 2; % we use 3 in our expeiments except this demo
    LCD_Len = 20;
    
    for iter = 1:outterIter
        if flagMRLR == 1
            y_w = double(y(:))/norm(double(y(:)));
            c = abs(D'*y_w);
            c = exp(c./sigma);
            c = max(c)-c;
            [y, transformation] = MRLR(I0, I0x, I0y, D,...
                                               c, imgSize, transformation,...
                                               transformType, frmCoords,...
                                               innerIter, display);
              
        elseif flagMRLR == 2
            y_w = double(y(:))/(norm(double(y(:))+eps));
            c = abs(D'*y_w);
            [~, Ix]  =    sort(c, 'descend');
            LCD      =    D(:, Ix(1:LCD_Len));
            LCD      =    LCD * diag(1./sqrt(sum(LCD.*LCD))); % normalize each atoms
            [y, transformation] = MRLR(I0, I0x, I0y, LCD,...
                                              zeros(LCD_Len, 1), imgSize,...
                                              transformation, transformType,...
                                              frmCoords, innerIter, display);
        end
    end
    if display == 1
        Tfm = fliptform(maketform('projective',inv(transformation')));
        frm = tformfwd(point2frm(frmCoords(1:2,:))', Tfm )';
        plot(frm(1,:),frm(2,:),'b-','LineWidth', 3);
    end
end

