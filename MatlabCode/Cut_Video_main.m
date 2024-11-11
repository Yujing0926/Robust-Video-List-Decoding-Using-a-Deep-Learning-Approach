clc
clear all
fclose('all');
tic

SYSTEM = "Windows";

switch SYSTEM
    case "Windows"
        APPEND_CHAR = '\\';  % Windows
    case "Linux"
        APPEND_CHAR = '/';  % Linux
    case "Mac"
        APPEND_CHAR = '/';  % Mac
end

%%% main route
MainResultRoute=['C:\Users\yzhang\Documents\YujingZHANG\CNN_Classification_Candidates-main' APPEND_CHAR];  
%%% your path to save the original videos for the Internet
AviRoute=[MainResultRoute '1080p_original_videos' APPEND_CHAR]; 

RawsequenceRoute=[MainResultRoute '1080p_rawSequence' APPEND_CHAR];
if (~exist(RawsequenceRoute))
    mkdir(RawsequenceRoute)
end

fileFolder=fullfile(AviRoute);
dirOutput_0=dir(fullfile(fileFolder,'*.yuv'));
dirOutput_1=dir(fullfile(fileFolder,'*.avi'));
dirOutput_2=dir(fullfile(fileFolder,'*.y4m'));
fileNames_0={dirOutput_0.name}';
fileNames_1={dirOutput_1.name}';
fileNames_2={dirOutput_2.name}';
fileNames=[fileNames_0;fileNames_1;fileNames_2];
videonames_0=strrep(fileNames_0,'.yuv','');
videonames_1=strrep(fileNames_1,'.avi','');
videonames_2=strrep(fileNames_2,'.y4m','');
videonames=[videonames_0; videonames_1; videonames_2];

for i = 1:size(videonames,1)

    SEQUENCE_NAME = videonames{i};
    
    if fileNames{i}(end-3:end) == '.avi'
        ConvertVideoCommand = append('ffmpeg -i ',AviRoute,fileNames{i},' -c:v libx264 -preset ultrafast -qp 0 ', RawsequenceRoute,videonames{i},'.mp4');
        % ConvertVideoCommand = append('ffmpeg -i ',AviRoute,fileNames{i},' -video_size 1920x1080 -c:v libx264 -preset ultrafast -qp 0 ',RawsequenceRoute,videonames{i},'.mp4');
        system(ConvertVideoCommand);
    elseif fileNames{i}(end-3:end) == '.yuv' 
        ConvertVideoCommand = append('ffmpeg -s 1920x1080 -i ',AviRoute,fileNames{i},' -c:v libx264 -preset ultrafast -qp 0 ', RawsequenceRoute,videonames{i},'.mp4');
        system(ConvertVideoCommand);
    elseif fileNames{i}(end-3:end) == '.y4m' 
        ConvertVideoCommand = append('ffmpeg -s 1920x1080 -i ',AviRoute,fileNames{i},' -c:v libx264 -preset ultrafast -qp 0 ', RawsequenceRoute,videonames{i},'.mp4');
        system(ConvertVideoCommand);
    end

    VideoRoute = append(MainResultRoute,APPEND_CHAR,'1080p_avi2yuv_videos',APPEND_CHAR,SEQUENCE_NAME,APPEND_CHAR,'HEVC',APPEND_CHAR,'encoding');
    if (~exist(VideoRoute))
        mkdir(VideoRoute)
    end

    PathVideo = append(RawsequenceRoute,APPEND_CHAR,SEQUENCE_NAME,'.mp4');

    VideoCutNameFile = append(SEQUENCE_NAME,'_','notresize');
    VideoCutNameFileResize = append(SEQUENCE_NAME);
    VideoYUVNameFile = append(SEQUENCE_NAME,'_1080p'); 

%   if exist(PathVideo, 'file') == 2
    video = VideoReader(PathVideo);
%   if (video.Height == 1920 && video.Width == 1080)

    %Cut Video
    CutVideoCommand = append('ffmpeg -i ',PathVideo,' ','-ss 00:00:00 -t 00:00:03 -async 1',' ', VideoRoute,APPEND_CHAR,VideoCutNameFile,'.mp4');
    system(CutVideoCommand);
    CropVideoCommand = append('ffmpeg -i ',VideoRoute,APPEND_CHAR,VideoCutNameFile,'.mp4',' -filter:v crop=1920:1024 ',VideoRoute,APPEND_CHAR,VideoCutNameFileResize,'.mp4');
    system(CropVideoCommand)
    
    %YUV file
    YUVCommand = append('ffmpeg -i ',VideoRoute,APPEND_CHAR,VideoCutNameFileResize,'.mp4',' -c:v rawvideo -pix_fmt yuv420p ',VideoRoute,APPEND_CHAR,VideoYUVNameFile,'.yuv');
    system(YUVCommand);

    %Copy file => RawSequence
    copyfile(append(VideoRoute,APPEND_CHAR,VideoYUVNameFile,'.yuv'),append(RawsequenceRoute,APPEND_CHAR,VideoYUVNameFile,'.yuv'));

    %Delete temporary video file
    DeleteCommandTemp = append('del',' ', PathVideo);
    system(DeleteCommandTemp);

    %Delete notresize video file
    DeleteCommandNotResize = append(VideoRoute,APPEND_CHAR, VideoCutNameFile,'.mp4');
    system(DeleteCommandNotResize);
    DeleteCommandTemp = append('del',' ', VideoRoute,APPEND_CHAR,VideoCutNameFileResize,'.mp4');
    system(DeleteCommandTemp);
    DeleteCommandYUV = append('del',' ', VideoRoute,APPEND_CHAR,VideoYUVNameFile,'.yuv');
    system(DeleteCommandYUV);
            
%   end
%   end
end

%% Radomly seperate the sequences for training and for inference
test_rate = 0.4;
num_samples = size(videonames,1);
indices = randperm(num_samples,round(test_rate*num_samples));

group_1_indices = indices(1:round(test_rate*num_samples));
group_2_indices = indices(round(test_rate*num_samples)+1:end);
group_1_name = videonames(group_1_indices,:);
group_2_name = videonames(group_2_indices,:);

TestRawsequenceRoute=[MainResultRoute APPEND_CHAR '1080p_rawSequence_test' APPEND_CHAR];
if (~exist(TestRawsequenceRoute))
    mkdir(TestRawsequenceRoute)
end

for j = 1:size(group_1_name,1)
    SEQUENCE_NAME = videonames{j};
    original_name = [RawsequenceRoute APPEND_CHAR SEQUENCE_NAME '_1080p.yuv'];
    original_name_2 = [TestRawsequenceRoute APPEND_CHAR SEQUENCE_NAME '.yuv'];
    copyfile(original_name, original_name_2);
    DeleteCommandYUV = append('del',' ', original_name);
    system(DeleteCommandYUV);
end
