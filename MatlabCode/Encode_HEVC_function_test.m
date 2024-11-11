function [] = Encode_HEVC_function_test(MainRouteResult,videonames,QPlist,APPEND_CHAR)
%% Inputs
% MainRouteResult='C:\Users\AR28470\Documents\Yujing_ZHANG\correction-crc-master'; % folder path
% 
% OriginalVideo=fullfile('C:\Users\AR28470\Documents\Yujing_ZHANG\correction-crc-master\1080p_rawSequence_test');
% 
% dirOutput=dir(fullfile(OriginalVideo,'*.yuv'));
% 
% fileNames={dirOutput.name}';
% videonames=strrep(fileNames,'.yuv','');

for i = 1:size(videonames,1)
% for i = 1
    
SEQUENCE_NAME = videonames{i}; % any name

cfgName='encoder_lowdelay_main'; %General coding parameter file name

OriginalYuvName = videonames{i}; % raw yuv seq name
ImageWidth = 1920; 
ImageHeight= 1024;  

NbCodedFrames = 10;

%% Other path
EncoderExeRoute= [MainRouteResult APPEND_CHAR 'MatlabCode' APPEND_CHAR 'EncoderHEVC'];
cfgRoute=MainRouteResult;
OriginalYuvRoute=[MainRouteResult APPEND_CHAR '1080p_rawSequence_test'];
OutRoute = [MainRouteResult APPEND_CHAR '1080p_videos_test' APPEND_CHAR];
if (~exist(OutRoute))
    mkdir(OutRoute);
end

OutRouteResult=[MainRouteResult APPEND_CHAR '1080p_videos_test' APPEND_CHAR SEQUENCE_NAME APPEND_CHAR 'HEVC' APPEND_CHAR 'encoding'];

%%
%NbCTUInSlice=ImageWidth/64;
NbCTUInSlice=3;
if (~exist([OutRouteResult APPEND_CHAR SEQUENCE_NAME  ]))
    mkdir([OutRouteResult APPEND_CHAR SEQUENCE_NAME ])
end
fileID = fopen([OutRouteResult APPEND_CHAR SEQUENCE_NAME '_encoding_info.txt'],'w');
fprintf(fileID,'%s \t\t %s \t\t %s \t\t %s\r\n','QP','Bitrate(kb/s)','NbMBInSlice','NbCodedFrames');
fprintf(fileID,'%s \r\n','*****************************************************************************');

%% copy original.yuv and  cfg files into Matlab Encoder Exe folder
copyfile([OriginalYuvRoute APPEND_CHAR OriginalYuvName '.yuv'  ],[EncoderExeRoute APPEND_CHAR OriginalYuvName '.yuv']);
copyfile([cfgRoute APPEND_CHAR cfgName '.cfg'  ],[EncoderExeRoute APPEND_CHAR cfgName '.cfg']);

cd([EncoderExeRoute]);

%% start encoding for each qp
for i=1: length(QPlist)
% for i = 1
    QP=QPlist(i)
    %mkdir([OUTPUT_DIR APPEND_CHAR SEQUENCE_NAME APPEND_CHAR 'qp' int2str(QP) ])
    
    LENCOD_EXE = [EncoderExeRoute APPEND_CHAR 'TAppEncoder.exe'];
    Command_string = [ '"' LENCOD_EXE '"' ]
    %% change cfg file
    Command_string=[Command_string ' -c "' cfgRoute APPEND_CHAR cfgName  '.cfg" '];
    Command_string=[Command_string ' -c "' cfgRoute APPEND_CHAR 'encoder_lowdelay_main.cfg" '];
    Command_string=[Command_string ' --InputFile="' OriginalYuvName '.yuv "'];
    Command_string=[Command_string ' --SourceWidth=' int2str(ImageWidth) ' --SourceHeight=' int2str(ImageHeight)];
    %Command_string=[Command_string ' --OutputWidth=' int2str(ImageWidth) ' --OutputHeight=' int2str(ImageHeight)];
    %Command_string=[Command_string ' --FrameSkip=' int2str(30) ];
    Command_string=[Command_string ' --FramesToBeEncoded=' int2str(NbCodedFrames) ];
    Command_string=[Command_string ' --BitstreamFile=' OutRouteResult APPEND_CHAR SEQUENCE_NAME '_qp' int2str(QP) '.265'];
    Command_string=[Command_string ' --ReconFile=' OutRouteResult APPEND_CHAR SEQUENCE_NAME '_qp' int2str(QP) '.yuv'];
    %Command_string=[Command_string ' -p TraceFile=\"' OutRouteResult APPEND_CHAR SEQUENCE_NAME '_qp' int2str(QP) '.trace.txt\"'];
    %Command_string=[Command_string ' -p StatsFile=\"' OutRouteResult APPEND_CHAR SEQUENCE_NAME '_qp' int2str(QP) '.stats.txt\"'];
    Command_string=[Command_string ' --QP=' int2str(QP) ];
    %Command_string=[Command_string '--IntraPeriod=30'];
    Command_string=[Command_string ' --FrameRate=30'];
    Command_string=[Command_string ' --SliceMode=0' ];
    Command_string=[Command_string ' --SliceArgument=1500' ];
    %Command_string=[Command_string ' -p IntraPeriod=0' ]; % only first I other P: IPPP...
    
    
    [status ExCode]=system(Command_string)
    
    if (status)
        error(['Command TAppEncoder did not complete successfully (return code ' int2str(status) '); Command was' Command_string]);
    end
    
    po=findstr(ExCode,'Bit rate (kbit/s)' );
    line=ExCode(1,po:end);
    po_1=findstr(line,':');
    po_2=findstr(line ,'Bits to avoid Startcode Emulation' ) ;
    bitrate_str=(line(1,po_1+1:po_2-1));
    bitrate=round(str2num(bitrate_str))
    
    fprintf(fileID,'%d \t\t %d \t\t\t %d \t\t\t %d\r\n',QP,bitrate,NbCTUInSlice,NbCodedFrames);
end
fprintf(fileID,'%s \r\n','*****************************************************************************');
fclose(fileID);

fclose('all')

%% delete exta file in EncoderExeRoute

delete([EncoderExeRoute APPEND_CHAR OriginalYuvName '.yuv']);
delete([EncoderExeRoute APPEND_CHAR cfgName '.cfg']);
delete([EncoderExeRoute APPEND_CHAR 'leakybucketparam.cfg']);
delete([EncoderExeRoute APPEND_CHAR 'log.dat']);
delete([EncoderExeRoute APPEND_CHAR 'data.txt']);

fprintf('Check the following route, delete extra files there \n');
fprintf('%s\n', EncoderExeRoute);
fprintf('done\n');
end
%
