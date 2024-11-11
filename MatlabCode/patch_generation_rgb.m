function patch_generation_rgb(OriginalYUVRoute,original_name,CandidateRoute,candidate_name,startfrm, ...
                              ImageWidth,ImageHeight,yuvformat, patch_size, patch_name,...
                              patch_route_rgb, file_route_rgb, APPEND_CHAR, type,img_candidate_route, ...
                              save_image,DCTT)
       
    if (~exist(patch_route_rgb))
        mkdir(patch_route_rgb)
    end
    
    if (~exist(file_route_rgb))
        mkdir(file_route_rgb)
    end
      
    video_ori_name = ['%s' APPEND_CHAR '%s' '.yuv'];
    video_ori = sprintf(video_ori_name,OriginalYUVRoute,original_name);
    video_dis_name = ['%s' APPEND_CHAR '%s' '.yuv'];
    video_dis = sprintf(video_dis_name,CandidateRoute,candidate_name);
    format = [ImageWidth ImageHeight];

    [Y_ori,U_ori,V_ori] = yuv_import(video_ori,format,1,startfrm-1,yuvformat);
    [Y_dis,U_dis,V_dis] = yuv_import(video_dis,format,1,startfrm-1,yuvformat);
    
    Y_ori = cell2mat(Y_ori);
    U_ori = cell2mat(U_ori);
    V_ori = cell2mat(V_ori);
    
    if yuvformat == "YUV420_8"
%         U_ori = imresize(U_ori, 2, 'bilinear');
%         V_ori = imresize(V_ori, 2, 'bilinear');
        a = 2;
        flt = [1 2 1; 2 4 2; 1 2 1]/4;
        U_ori = upsample(upsample(U_ori,a)',a)';
        U_ori = uint8(filter2(flt,flipud(fliplr(U_ori))));
        U_ori = flipud(fliplr(U_ori));
        U_ori(1:ImageHeight, ImageWidth) = U_ori(1:ImageHeight,ImageWidth-1);
        U_ori(ImageHeight, 1:ImageWidth) = U_ori(ImageHeight-1,1:ImageWidth);
        
        V_ori = upsample(upsample(V_ori,a)',a)';
        V_ori = uint8(filter2(flt,flipud(fliplr(V_ori))));
        V_ori = flipud(fliplr(V_ori)); 
        V_ori(1:ImageHeight, ImageWidth) = V_ori(1:ImageHeight,ImageWidth-1);
        V_ori(ImageHeight, 1:ImageWidth) = V_ori(ImageHeight-1,1:ImageWidth);
    end
    
%     YUV_ori = cat(3, Y_ori, U_ori, V_ori);

    Y_dis = cell2mat(Y_dis);
    U_dis = cell2mat(U_dis);
    V_dis = cell2mat(V_dis);
    
    if yuvformat == "YUV420_8"
        % sample repeat interpolation 
        a = 2;
        flt = [1 1 ; 1 1];
        
        Y444 = Y_dis;
        Cbu   = upsample(upsample(U_dis,a)',a)';
        Cb444 = uint8(filter2(flt,flipud(fliplr(Cbu))));
        Cb444 = flipud(fliplr(Cb444));
        
        Cru   = upsample(upsample(V_dis,a)',a)';
        Cr444 = uint8(filter2(flt,flipud(fliplr(Cru))));
        Cr444 = flipud(fliplr(Cr444));   

        % bilinear interpolation
%         U_dis(1:4,1:4)=[135   135   135   135;
%                        165   130   100   134;
%                        80   75   0   132;
%                        131   131   131   131];
%         U_dis(1:4,1:4)
%         U_dis = imresize(U_dis, 2, 'bilinear');
%         V_dis = imresize(V_dis, 2, 'bilinear');
        a = 2;
        flt = [1 2 1; 2 4 2; 1 2 1]/4;
        U_dis = upsample(upsample(U_dis,a)',a)';
%         round(U_dis(1:8,1:8))
        U_dis = uint8(filter2(flt,flipud(fliplr(U_dis))));
        U_dis = flipud(fliplr(U_dis));
%         U_dis = imfilter(U_dis, flt);
        U_dis(1:ImageHeight, ImageWidth) = U_dis(1:ImageHeight,ImageWidth-1);
        U_dis(ImageHeight, 1:ImageWidth) = U_dis(ImageHeight-1,1:ImageWidth);
        
        V_dis = upsample(upsample(V_dis,a)',a)';
        V_dis = uint8(filter2(flt,flipud(fliplr(V_dis))));
        V_dis = flipud(fliplr(V_dis)); 
%         V_dis = imfilter(V_dis, flt);
        V_dis(1:ImageHeight, ImageWidth) = V_dis(1:ImageHeight,ImageWidth-1);
        V_dis(ImageHeight, 1:ImageWidth) = V_dis(ImageHeight-1,1:ImageWidth);
        
%         round(U_dis(1:8,1:8))
    end
    
    YUV_dis = cat(3, Y_dis, U_dis, V_dis);
    RGB_img = ycbcr2rgb(uint8(YUV_dis));

    if type == "train"
        fileIDPSNR_rgb = fopen([file_route_rgb APPEND_CHAR patch_name '_PSNR_VALUE_pattern.txt'], "wt");
        fileIDSSIM_rgb = fopen([file_route_rgb APPEND_CHAR patch_name '_SSIM_VALUE_pattern.txt'], "wt");
    end
    
    M = patch_size;
    N = patch_size;

    xpatch =ImageHeight/M;
    ypatch =ImageWidth/N;

    for x = 1 : xpatch
        for y = 1 : ypatch
            
            i = (x-1) * (M) ;
            j = (y-1) * (N) ;
            st_idx=(i+1);
            st_idy=(j+1);
            patch_original_y = Y_ori(st_idx:i + M, st_idy:j + N);
            patch_original_u = U_ori(st_idx:i + M, st_idy:j + N);
            patch_original_v = V_ori(st_idx:i + M, st_idy:j + N);
            patch_original = cat(3, patch_original_y, patch_original_u, patch_original_v);
            
            
            patch_modified_y = Y_dis(st_idx:i + M, st_idy:j + N);
            patch_modified_u = U_dis(st_idx:i + M, st_idy:j + N);
            patch_modified_v = V_dis(st_idx:i + M, st_idy:j + N);
            patch_modified = cat(3, patch_modified_y, patch_modified_u, patch_modified_v);
            
%             distorImageNa.Y = mean(mean((double(patch_original_y) - double(patch_modified_y)).^2));
%             distorImageNa.U = mean(mean((double(patch_original_u) - double(patch_modified_u)).^2));
%             distorImageNa.V = mean(mean((double(patch_original_v) - double(patch_modified_v)).^2));

            %%% in order to separate green ang black patchs in rgb format
            RGB_dis = ycbcr2rgb(uint8(patch_modified));
            RGB_ori = ycbcr2rgb(uint8(patch_original));
            
%             if all(patch_modified_y==0) & all(patch_modified_u==0) & all(patch_modified_v==0)
%                 % RGB_dis(:,:,2) = 0;
%                 for k = 1:M
%                     for l = 1:N
%                         RGB_dis(k,l,1) = 255*(((-1)^(st_idx+k+st_idy+l)+1)/2);
%                         RGB_dis(k,l,2) = 255*(((-1)^(st_idx+k)+1)/2);
%                         RGB_dis(k,l,3) = 255*(((-1)^(st_idy+l)+1)/2);
%                     end
%                 end
% 
%             end

%% Add Discriminant Color Texture Transformation
            if DCTT == 0
                nb=0;
                for k = 1:M
                    done = 0;
                    for l = 1:N
                        if (Y444(i+k,j+l)==0 & Cb444(i+k,j+l)==0 & Cr444(i+k,j+l)==0)
    %                         RGB_dis(k,l,1)=((-1)^(i+k+j+l)+1)/2*255;
    %                         RGB_dis(k,l,2)=((-1)^(j+l)+1)/2*255;
    %                         RGB_dis(k,l,3)=((-1)^(i+k)+1)/2*255;
    % %                         fprintf('(%d,%d) ',i+k-1,j+l-1);
                            done=1;
                            nb=nb+1;
                        end
                    end
    %                 if done
    %                     fprintf('\n');
    %                 end
                end
    %             fprintf('Nb green pixels = %d\n',nb);
    
                if nb ~= 0
                    for k = 1:M
                        for l = 1:N
                            if (Y444(i+k,j+l)==0 & Cb444(i+k,j+l)==0 & Cr444(i+k,j+l)==0)
                                RGB_dis(k,l,1)=((-1)^(i+k+j+l)+1)/2*255;
                                RGB_dis(k,l,2)=((-1)^(j+l)+1)/2*255;
                                RGB_dis(k,l,3)=((-1)^(i+k)+1)/2*255;
                                RGB_img(i+k,j+l,1)=((-1)^(i+k+j+l)+1)/2*255;
                                RGB_img(i+k,j+l,2)=((-1)^(j+l)+1)/2*255;
                                RGB_img(i+k,j+l,3)=((-1)^(i+k)+1)/2*255;
                            end
                        end
                    end
                end
            end
            
            if startfrm == 1
                image_dis_name = ['%s' APPEND_CHAR 'patch_%s_' num2str(x-1) '_' num2str(y-1) '.png' ];
            elseif startfrm == 2
                image_dis_name = ['%s' APPEND_CHAR 'patch_%s_' num2str(x-1) '_' num2str(y-1) '.png' ];
            end
            image_dis_name = sprintf(image_dis_name,patch_route_rgb,patch_name);
            imwrite(RGB_dis,image_dis_name);
            
            %% calculate psnr and ssim of the rgb patchs
            if type == "train"
                psnr_rgb = psnr(RGB_ori, RGB_dis);
                ssim_rgb = ssim(RGB_ori, RGB_dis);
                if startfrm == 1
                    fprintf(fileIDPSNR_rgb,'patch_%s_%s_%s=%.2f\n',patch_name, num2str(x-1), num2str(y-1),psnr_rgb);
                    fprintf(fileIDSSIM_rgb,'patch_%s_%s_%s=%.2f\n',patch_name, num2str(x-1), num2str(y-1),ssim_rgb);
                elseif startfrm == 2
                    fprintf(fileIDPSNR_rgb,'interpatch_%s_%s_%s=%.2f\n',patch_name, num2str(x-1), num2str(y-1),psnr_rgb);
                    fprintf(fileIDSSIM_rgb,'interpatch_%s_%s_%s=%.2f\n',patch_name, num2str(x-1), num2str(y-1),ssim_rgb);
                end
            end
            
        end
    end
    
    if type == "train"
        fclose(fileIDPSNR_rgb);
        fclose(fileIDSSIM_rgb);
    end

%% Save the image with RGB error pattern
if save_image == 0
    image_name = append("%s",APPEND_CHAR,"%s.png");
    image_name = sprintf(image_name,img_candidate_route, patch_name);
    imwrite(RGB_img,image_name);
end

end


