function save_yuvframe(ImgResultRoute,CandidateRoute,candidate_name,startfrm,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR)

    if (~exist(ImgResultRoute))
        mkdir(ImgResultRoute)
    end

    video_dis_name = ['%s' APPEND_CHAR '%s' '.yuv'];
    video_dis = sprintf(video_dis_name,CandidateRoute,candidate_name);
    format = [ImageWidth ImageHeight];

    [Y_dis,U_dis,V_dis] = yuv_import(video_dis,format,1,startfrm-1,yuvformat);

    Y_dis = cell2mat(Y_dis);
    U_dis = cell2mat(U_dis);
    V_dis = cell2mat(V_dis);

    % Need to upsample the chroma to convert into RGB
    U_dis = imresize(U_dis, 2, 'bilinear');
    V_dis = imresize(V_dis, 2, 'bilinear');
    YUV_dis = cat(3, Y_dis, U_dis, V_dis);

    %Convert YUV to RGB (MATLAB function ycbcr2rgb uses BT.601 conversion formula).
    RGB_dis = ycbcr2rgb(uint8(YUV_dis));
    
    %%% in order to separate green ang black patchs
%     if all(Y_dis==0) & all(U_dis==0) & all(V_dis==0)
%         RGB_dis(:,:,2) = 0;
%     end
    
    % Save the images 
    image_dis_name = ['%s' APPEND_CHAR '%s' '_Im_' num2str(startfrm) '_.png' ];
    image_dis_name = sprintf(image_dis_name,ImgResultRoute, candidate_name);

    imwrite(RGB_dis,image_dis_name);

end
