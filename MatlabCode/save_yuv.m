function save_yuv(ImgResultRoute,CandidateRoute,candidate_name,startfrm,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR)

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
    
    U_dis = imresize(U_dis, 2, 'bilinear');
    V_dis = imresize(V_dis, 2, 'bilinear');
    YUV_dis = cat(3, Y_dis, U_dis, V_dis);
    
    filename = append(ImgResultRoute, APPEND_CHAR, candidate_name, "_Im_", num2str(startfrm), "_Y.yuv");
    fid=fopen(filename,'w');
    count = fwrite(fid,Y_dis','ubit8');
    fclose(fid)

    filename = append(ImgResultRoute, APPEND_CHAR, candidate_name, "_Im_", num2str(startfrm), "_U.yuv");
    fid=fopen(filename,'w');
    count = fwrite(fid,U_dis','ubit8');
    fclose(fid)

    filename = append(ImgResultRoute, APPEND_CHAR, candidate_name, "_Im_", num2str(startfrm), "_V.yuv");
    fid=fopen(filename,'w');
    count = fwrite(fid,V_dis','ubit8');
    fclose(fid)
    
    filename = append(ImgResultRoute, APPEND_CHAR, candidate_name, "_Im_", num2str(startfrm), ".yuv");
    fid=fopen(filename,'w');
    count = fwrite(fid,Y_dis','ubit8');
    count = fwrite(fid,U_dis','ubit8');
    count = fwrite(fid,V_dis','ubit8');
    fclose(fid)

%     % Need to upsample the chroma to convert into RGB
%     U_dis = imresize(U_dis, 2, 'bilinear');
%     V_dis = imresize(V_dis, 2, 'bilinear');
%     YUV_dis = cat(3, Y_dis, U_dis, V_dis);
% 
%     %Convert YUV to RGB (MATLAB function ycbcr2rgb uses BT.601 conversion formula).
%     RGB_dis = ycbcr2rgb(uint8(YUV_dis));
% 
%     % Save the images 
%     image_dis_name = ['%s' APPEND_CHAR '%s' '_Im_' num2str(startfrm) '_.png' ];
%     image_dis_name = sprintf(image_dis_name,ImgResultRoute, candidate_name);
% 
%     imwrite(RGB_dis,image_dis_name);

end