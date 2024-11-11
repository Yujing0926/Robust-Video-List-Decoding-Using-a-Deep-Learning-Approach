function [psnr_score,ssim_score]=PSNR_YUV_Generation(YUV1Name,YUV1Route,YUV2Name,YUV2Route,ImageHeight,ImageWidth,error_frame,APPEND_CHAR)

video_ori = sprintf('%s\\%s.yuv',YUV1Route, YUV1Name);
video_dis = append(YUV2Route,APPEND_CHAR,YUV2Name,'.yuv');
video_dis = char(video_dis);
format = [ImageWidth ImageHeight];

[Y_ori,U_ori,V_ori] = yuv_import(video_ori,format,1,error_frame-1,format);
[Y_dis,U_dis,V_dis] = yuv_import(video_dis,format,1,error_frame-1,format);

Y_dis = cell2mat(Y_dis);
U_dis = cell2mat(U_dis);
V_dis = cell2mat(V_dis);

Y_ori = cell2mat(Y_ori);
U_ori = cell2mat(U_ori);
V_ori = cell2mat(V_ori);

psnr_score=zeros(4,1);
ssim_score=zeros(4,1);

distorImageNa.Y = mean(mean((double(Y_ori) - double(Y_dis)).^2));
distorImageNa.U = mean(mean((double(U_ori) - double(U_dis)).^2));
distorImageNa.V = mean(mean((double(V_ori) - double(V_dis)).^2));

PSNR_Y = 10*log10(255^2./distorImageNa.Y');
PSNR_U = 10*log10(255^2./distorImageNa.U');
PSNR_V = 10*log10(255^2./distorImageNa.V');
PSNR_YUV = (6*PSNR_Y + PSNR_U + PSNR_V)/8;

psnr_score(1) = PSNR_Y;
psnr_score(2) = PSNR_U;
psnr_score(3) = PSNR_V;
psnr_score(4) = PSNR_YUV;

SSIM_Y = ssim(double(Y_ori),double(Y_dis));
SSIM_U = ssim(double(U_ori),double(U_dis));
SSIM_V = ssim(double(V_ori),double(V_dis));

U_dis = imresize(U_dis, 2, 'bilinear');
V_dis = imresize(V_dis, 2, 'bilinear');
YUV_dis = cat(3, Y_dis, U_dis, V_dis);

U_ori = imresize(U_ori, 2, 'bilinear');
V_ori = imresize(V_ori, 2, 'bilinear');
YUV_ori = cat(3, Y_ori, U_ori, V_ori);

SSIM_YUV = ssim(double(YUV_ori),double(YUV_dis));

ssim_score(1) = SSIM_Y;
ssim_score(2) = SSIM_U;
ssim_score(3) = SSIM_V;
ssim_score(4) = SSIM_YUV;


end