clc
clear all
fclose('all');
tic

global diffYuvSize;

SYSTEM = "Windows";

switch SYSTEM
    case "Windows"
        APPEND_CHAR = '\\';  % Windows
    case "Linux"
        APPEND_CHAR = '/';  % Linux
    case "Mac"
        APPEND_CHAR = '/';  % Mac
end

MainResultRoute=['E:\doctor\codes\CNN_Classification_Candidates' APPEND_CHAR];
video_route = MainResultRoute;
type = "train";
Codec = "HEVC";
frame = "intra";
PacketNum = 4; % choose the I frame or P frame wanted to corrupt
               % for HEVC, the first 3 packets are the Header packets, 
               % we should not change the Header packets.

yuvformat='YUV420_8';
ImageWidth = 1920; 
ImageHeight = 1024; 
format = [ImageWidth ImageHeight];
QPlist = [37]; %32 27 22
Total_Frame=10;
frameStart=0;
WR='WorkRoute';
SR='Simulation_Results';
diffYuvSize=0;

patch_size = 32;
patch_type = 'rgb';
generate_patch = 1; % 0: generate, 1: not generate
generate_super_patch = 0; % 0: generate, 1: not generate
save_image = 0; % save the image with RGB error pattern
DCTT = 0; % use Discriminant Color Texture Transformation color space conversion
          % 0: use, 1: not use

switch type
    case "train"       
        damage_rate_list = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99];  % to define the flipped bit position
        switch PacketNum
            case 4
                OriginalYUVRoute = [video_route APPEND_CHAR '1080p_rawSequence' APPEND_CHAR];
                videoname_route = [video_route APPEND_CHAR 'Candidates_1024p_train_intra'];
            case 5
                OriginalYUVRoute = [video_route APPEND_CHAR '1080p_rawSequence' APPEND_CHAR];
                videoname_route = [video_route APPEND_CHAR 'Candidates_1024p_train_inter'];
        end
        fileFolder=fullfile(OriginalYUVRoute);
        dirOutput=dir(fullfile(fileFolder,'*.yuv'));
        fileNames={dirOutput.name}';
        videonames=strrep(fileNames,'.yuv','');
        %%% encode the original video with 4 different QPs
        Encode_HEVC_function(MainResultRoute,videonames,QPlist,APPEND_CHAR)
    
    case "test"
%         damage_rate_list = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99];
        damage_rate_list = [0.3 0.6 0.9];
        switch PacketNum 
            case 4
                OriginalYUVRoute = [video_route APPEND_CHAR '1080p_rawSequence_test' APPEND_CHAR];
                videoname_route = [video_route APPEND_CHAR 'Candidates_1024p_test_intra'];
            case 5
                OriginalYUVRoute = [video_route APPEND_CHAR '1080p_rawSequence_test' APPEND_CHAR];
                videoname_route = [video_route APPEND_CHAR 'Candidates_1024p_test_inter'];
        end
        fileFolder=fullfile(OriginalYUVRoute);
        dirOutput=dir(fullfile(fileFolder,'*.yuv'));
        fileNames={dirOutput.name}';
        videonames=strrep(fileNames,'.yuv','');
        Encode_HEVC_function_test(MainResultRoute,videonames,QPlist,APPEND_CHAR)
end

if (~exist(OriginalYUVRoute))
    mkdir(OriginalYUVRoute)
end
if (~exist(videoname_route))
    mkdir(videoname_route)
end


for i = 1:size(videonames,1)
    SEQUENCE_NAME = videonames{i};
           
%     for j = 1:size(QPlist,2)
        for j = 1
            
            Qp = QPlist(j);
                        
            if Codec=="HEVC"
                switch PacketNum
                    case 4
                        if type == "train"
                            CandidatesRoute = [video_route APPEND_CHAR 'Candidates_1024p_train_intra' APPEND_CHAR SEQUENCE_NAME APPEND_CHAR 'qp' num2str(Qp)];
                            database_route = [MainResultRoute APPEND_CHAR '1024p_database_patchs' num2str(patch_size) '_train_intra'];
                        else
                            CandidatesRoute = [video_route APPEND_CHAR 'Candidates_1024p_test_intra' APPEND_CHAR SEQUENCE_NAME APPEND_CHAR 'qp' num2str(Qp)];
                            database_route = [MainResultRoute APPEND_CHAR '1024p_database_patchs' num2str(patch_size) '_test_intra'];
                        end
                    case 5
                        if type == "train"
                            CandidatesRoute = [video_route APPEND_CHAR 'Candidates_1024p_train_inter' APPEND_CHAR SEQUENCE_NAME APPEND_CHAR 'qp' num2str(Qp)];
                            database_route = [MainResultRoute APPEND_CHAR '1024p_database_patchs' num2str(patch_size) '_train_inter'];
                        else
                            CandidatesRoute = [video_route APPEND_CHAR 'Candidates_1024p_test_inter' APPEND_CHAR SEQUENCE_NAME APPEND_CHAR 'qp' num2str(Qp)];
                            database_route = [MainResultRoute APPEND_CHAR '1024p_database_patchs' num2str(patch_size) '_test_inter'];
                        end
                end
            DecoderRoute=[MainResultRoute APPEND_CHAR 'MatlabCode' APPEND_CHAR 'DecoderHEVC'];
            CodeRoute=[MainResultRoute APPEND_CHAR 'MatlabCode'];
            end

            %% make folder for work
            if (~exist(CandidatesRoute))
                mkdir(CandidatesRoute)
            end
            
            %% Intact video
            Error_frame = PacketNum-3; 
            OriginalYUVName = videonames{i}; %[videonames{i} '_' num2str(Error_frame)]; 
            In265Name=[SEQUENCE_NAME '_qp' num2str(Qp)];
            IntactYuvName=['dec_' In265Name '_Intact']; %_Im_' num2str(Error_frame)
            
            switch type
                case "train"
                    In265Route=[video_route APPEND_CHAR '1080p_videos' APPEND_CHAR SEQUENCE_NAME APPEND_CHAR 'HEVC' APPEND_CHAR 'encoding'];               
                    IntactRoute=append(video_route, APPEND_CHAR, '1080p_videos', APPEND_CHAR, SEQUENCE_NAME, APPEND_CHAR, ...
                        'HEVC',APPEND_CHAR,'decoding',APPEND_CHAR,'qp', num2str(Qp), APPEND_CHAR, SR, APPEND_CHAR,'Intact');
                    if generate_super_patch == 0
                        patch_route_rgb = [database_route APPEND_CHAR 'qp' num2str(Qp) '_sp' num2str(patch_size) '_' patch_type];
                    else
                        patch_route_rgb = [database_route APPEND_CHAR 'qp' num2str(Qp) '_patch' num2str(patch_size) '_' patch_type];
                    end
                    file_route_rgb = [database_route APPEND_CHAR  'patch' num2str(patch_size) '_psnr_qp' num2str(Qp) '_' patch_type APPEND_CHAR SEQUENCE_NAME];
                    img_candidate_route = [database_route APPEND_CHAR 'qp' num2str(Qp) '_' patch_type];

                case "test"
                    In265Route=[video_route APPEND_CHAR '1080p_videos_test' APPEND_CHAR SEQUENCE_NAME APPEND_CHAR 'HEVC' APPEND_CHAR 'encoding'];               
                    IntactRoute=append(video_route, APPEND_CHAR, '1080p_videos_test', APPEND_CHAR, SEQUENCE_NAME, APPEND_CHAR, ...
                        'HEVC',APPEND_CHAR,'decoding',APPEND_CHAR,'qp', num2str(Qp), APPEND_CHAR, SR, APPEND_CHAR,'Intact');                 
                    patch_route_rgb = [database_route APPEND_CHAR 'qp' num2str(Qp) '_' patch_type APPEND_CHAR SEQUENCE_NAME '_' num2str(Error_frame) APPEND_CHAR 'i' num2str(i-1)];
                    file_route_rgb = [database_route APPEND_CHAR 'qp' num2str(Qp) '_' patch_type APPEND_CHAR SEQUENCE_NAME '_' num2str(Error_frame) APPEND_CHAR];
                    img_candidate_route = [database_route APPEND_CHAR 'qp' num2str(Qp) '_' patch_type APPEND_CHAR SEQUENCE_NAME '_' num2str(Error_frame) APPEND_CHAR];
            end
            if (~exist(IntactRoute))
                mkdir(IntactRoute)
            end
            if (~exist(patch_route_rgb))
                mkdir(patch_route_rgb)
            end
            if (~exist(file_route_rgb))
                mkdir(file_route_rgb)
            end
            if (~exist(img_candidate_route))
                mkdir(img_candidate_route)
            end
            
            original_frame = frameStart + Error_frame;
            Type='Intact';
            cd(CodeRoute);
            [d text IntactYuvName]=...
                Decode_from_matlab_HEVC_Function(In265Name,In265Route,DecoderRoute,IntactRoute,Type,APPEND_CHAR);
            cd(CodeRoute);
             
            save_yuvframe(OriginalYUVRoute,OriginalYUVRoute,OriginalYUVName,original_frame,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR);
%             save_yuv(IntactRoute,IntactRoute,IntactYuvName,Error_frame,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR);
            save_yuvframe(IntactRoute,IntactRoute,IntactYuvName,Error_frame,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR);            

            intact_name = [IntactYuvName '_Im_' num2str(Error_frame)];
            original_name = [OriginalYUVName '_Im_' num2str(Error_frame)];
            
            patch_name = ['i' num2str(i-1)];

            intact_img = append(IntactRoute, APPEND_CHAR, intact_name, "_.png");
            intact_img_copy = append(database_route, APPEND_CHAR, 'qp', num2str(Qp), '_', patch_type, APPEND_CHAR, ...
                SEQUENCE_NAME, '_', num2str(Error_frame), APPEND_CHAR, patch_name, ".png");
            
            if type == "train"
                 intact_img_copy = append(database_route, APPEND_CHAR, 'qp', num2str(Qp), '_', patch_type, APPEND_CHAR, patch_name, ".png");
            end
            if type == "test"
                 intact_img_copy = append(database_route, APPEND_CHAR, 'qp', num2str(Qp), '_', patch_type, APPEND_CHAR, ...
                SEQUENCE_NAME, '_', num2str(Error_frame), APPEND_CHAR, patch_name, ".png");
            end
            copyfile(intact_img, intact_img_copy);

            if generate_patch == 0                
                patch_generation_rgb(OriginalYUVRoute,OriginalYUVName,IntactRoute,IntactYuvName,Error_frame,...
                                         ImageWidth,ImageHeight,yuvformat,patch_size,patch_name,...
                                         patch_route_rgb, file_route_rgb, APPEND_CHAR, type, img_candidate_route, ...
                                         save_image,DCTT);
            end
            if generate_super_patch == 0                
                super_patch_generation_rgb(OriginalYUVRoute,OriginalYUVName,IntactRoute,IntactYuvName,Error_frame,...
                                         ImageWidth,ImageHeight,yuvformat,patch_size,patch_name,...
                                         patch_route_rgb, file_route_rgb, APPEND_CHAR, type, img_candidate_route, ...
                                         save_image, DCTT);
            end

            %% PSNR and SSIM
            [PsnrIntact,SSIMIntact]=PSNR_YUV_Generation...
                 (OriginalYUVName,OriginalYUVRoute,IntactYuvName,IntactRoute,ImageHeight,ImageWidth,Error_frame,APPEND_CHAR);
            
            PSNRIntact_Y = PsnrIntact(1,1);
            PSNRIntact_U = PsnrIntact(2,1);
            PSNRIntact_V = PsnrIntact(3,1);
            PSNRIntact_YUV = PsnrIntact(4,1);
            
            SSIMIntact_Y = SSIMIntact(1,1);
            SSIMIntact_U = SSIMIntact(2,1);
            SSIMIntact_V = SSIMIntact(3,1);
            SSIMIntact_YUV = abs(SSIMIntact(4,1));

            original_image = imread([OriginalYUVRoute APPEND_CHAR original_name '_.png']);
            intact_image = imread([IntactRoute APPEND_CHAR intact_name '_.png']);
            PSNRIntact_RGB = psnr(original_image,intact_image);
            SSIMIntact_RGB = ssim(original_image,intact_image);
            
            switch PacketNum
                case 4
                    PSNRIntactfile = append(IntactRoute,APPEND_CHAR,'MetricValues_intra.txt');
                    fileIDPSNR = fopen(PSNRIntactfile,'w');
                case 5
                    PSNRIntactfile = append(IntactRoute,APPEND_CHAR,'MetricValues.txt');
                    fileIDPSNR = fopen(PSNRIntactfile,'w');
%                     fileIDPSNR = fopen([IntactRoute '\MetricValues_yuv_padded.txt'],'wt');
            end
            
            fprintf(fileIDPSNR,'Error_frame_PSNR_RGB=%.2f\n',PSNRIntact_RGB);
            fprintf(fileIDPSNR,'Error_frame_PSNR_Y=%.2f\n',PSNRIntact_Y);
            fprintf(fileIDPSNR,'Error_frame_PSNR_U=%.2f\n',PSNRIntact_U);
            fprintf(fileIDPSNR,'Error_frame_PSNR_V=%.2f\n',PSNRIntact_V);
            fprintf(fileIDPSNR,'Error_frame_PSNR_YUV=%.2f\n',PSNRIntact_YUV);

            fprintf(fileIDPSNR,'Error_frame_SSIM_RGB=%.2f\n',SSIMIntact_RGB);
            fprintf(fileIDPSNR,'Error_frame_SSIM_Y=%.2f\n',SSIMIntact_Y);
            fprintf(fileIDPSNR,'Error_frame_SSIM_U=%.2f\n',SSIMIntact_U);
            fprintf(fileIDPSNR,'Error_frame_SSIM_V=%.2f\n',SSIMIntact_V);
            fprintf(fileIDPSNR,'Error_frame_SSIM_YUV=%.2f\n',SSIMIntact_YUV);

            fclose(fileIDPSNR);
        
            CandidateResultRoute=append(MainResultRoute,"Candidates_1024p_",type,"_",frame,APPEND_CHAR,SEQUENCE_NAME,APPEND_CHAR,"qp",num2str(Qp));
            PSNRerror_RGB = append(CandidateResultRoute, APPEND_CHAR, 'ErrorFramePSNRValues_RGB.txt');
            SSIMerror_RGB = append(CandidateResultRoute, APPEND_CHAR, 'ErrorFrameSSIMValues_RGB.txt');
    
            %% Generate the candidates of corrupted images
            for k = 1:size(damage_rate_list,2)
                damage_rate = damage_rate_list(k);  
                           
                %% generate the corrupted videos by fliping 1 bit in the first video packet
                if Codec=="HEVC"
                    cd(CodeRoute);
                    CandidatesGeneration(SEQUENCE_NAME, Qp,damage_rate,PacketNum,CandidatesRoute,DecoderRoute,ImageWidth,ImageHeight,yuvformat,CodeRoute,In265Route,APPEND_CHAR);
                end
    
                if Codec == "HEVC"
                    Error_frame = PacketNum-3;
                    original_name = [OriginalYUVName '_Im_' num2str(Error_frame)];
    
                    %% Extract corrupted image candidates and generate the patchs           
                    candidate_name = [SEQUENCE_NAME '_qp' num2str(Qp) '_frame_' num2str(Error_frame) '_damaged_' num2str(damage_rate)];
                    switch type
                        case "train"
                            candidate_name_new = ['c' num2str(i-1) '_' num2str(k-1)];
                        case "test"
                            candidate_name_new = ['c' num2str(i-1) '_' num2str(k-1)];
                    end
                    
    %                 save_yuv(CandidateResultRoute,CandidateResultRoute,candidate_name,Error_frame,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR);
    %                 save_yuvframe(CandidateResultRoute,CandidateResultRoute,candidate_name,Error_frame,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR);
    %                 save_name = [SEQUENCE_NAME '_qp' num2str(Qp) '_frame_' num2str(Error_frame) '_damaged_' num2str(damage_rate)];                
    %                 save_yuv(CandidatesRoute,CandidatesRoute,save_name,Error_frame,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR);                             
    %                 CandidateResultRoute = char(CandidateResultRoute);
                    
                    patch_name = ['c' num2str(i-1) '_' num2str(k-1)];               
                    switch type
                        case "train"
                            if generate_super_patch == 0
                                patch_route_rgb = [database_route APPEND_CHAR 'qp' num2str(Qp) '_sp' num2str(patch_size) '_' patch_type];
                            else
                                patch_route_rgb = [database_route APPEND_CHAR 'qp' num2str(Qp) '_patch' num2str(patch_size) '_' patch_type];
                            end
                            file_route_rgb = [database_route APPEND_CHAR  'patch' num2str(patch_size) '_psnr_qp' num2str(Qp) '_' patch_type APPEND_CHAR SEQUENCE_NAME];
                        case "test"
                            patch_route_rgb = [database_route APPEND_CHAR 'qp' num2str(Qp) '_' patch_type APPEND_CHAR SEQUENCE_NAME '_' num2str(Error_frame) APPEND_CHAR 'c' num2str(i-1) '_' num2str(k-1)];
                            file_route_rgb = [database_route APPEND_CHAR 'qp' num2str(Qp) '_' patch_type APPEND_CHAR SEQUENCE_NAME '_' num2str(Error_frame) APPEND_CHAR];
                    end
                    
                    candidate_img = append(CandidatesRoute, APPEND_CHAR, candidate_name, "_Im_", num2str(Error_frame), "_.png");
                    
                    if type == "train"
                       candidate_img_copy = append(database_route, APPEND_CHAR, 'qp', num2str(Qp), '_', patch_type, APPEND_CHAR, patch_name, ".png"); 
                    end
                    if type == "test"
                       candidate_img_copy = append(database_route, APPEND_CHAR, 'qp', num2str(Qp), '_', patch_type, APPEND_CHAR, SEQUENCE_NAME, '_', num2str(Error_frame), APPEND_CHAR, patch_name, ".png"); 
                    end
                    copyfile(candidate_img, candidate_img_copy);
    
                    fileIDPSNRerror_RGB = fopen(PSNRerror_RGB,'a+');
                    fileIDSSIMerror_RGB = fopen(SSIMerror_RGB,'a+'); 
                    original_image = imread([OriginalYUVRoute APPEND_CHAR original_name '_.png']);
                    candidate_image = imread(candidate_img);
                    PSNRCandidate_RGB = psnr(original_image,candidate_image);
                    SSIMCandidate_RGB = ssim(original_image,candidate_image);
                    fprintf(fileIDPSNRerror_RGB,'Error_frame_PSNR_RGB_%.1f=%.2f\n',damage_rate,PSNRCandidate_RGB);
                    fprintf(fileIDSSIMerror_RGB,'Error_frame_SSIM_RGB_%.1f=%.2f\n',damage_rate,SSIMCandidate_RGB);
                    fclose(fileIDPSNRerror_RGB);
                    fclose(fileIDSSIMerror_RGB);
    
                    if generate_patch == 0                
                        patch_generation_rgb(OriginalYUVRoute,OriginalYUVName,CandidatesRoute,candidate_name,Error_frame,...
                                                 ImageWidth,ImageHeight,yuvformat,patch_size,patch_name,...
                                                 patch_route_rgb, file_route_rgb, APPEND_CHAR, type, img_candidate_route, ...
                                                 save_image,DCTT);
                    end
                    if  generate_super_patch == 0
                        super_patch_generation_rgb(OriginalYUVRoute,original_name,CandidatesRoute,candidate_name,Error_frame,...
                                                 ImageWidth,ImageHeight,yuvformat,patch_size,patch_name,...
                                                 patch_route_rgb, file_route_rgb, APPEND_CHAR, type, img_candidate_route, ...
                                                 save_image,DCTT);
                    end   
                end           
            end
        end
end
