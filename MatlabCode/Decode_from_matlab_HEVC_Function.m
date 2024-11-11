%% 2015-06-05
%% decode seq by matlab
%  decoder exe file are here:
%~\matlab_code\Decoder

function [d text OutYuv ]=Decode_from_matlab_HEVC_Function(In265Name,In265Route,DecoderRoute,OutRoute,Type,APPEND_CHAR)

%% Input
%tic
% In264Name='basketballdrive_qp37';
% In264Route='E:\2015,s10,summer\PercentageOfSynchBits\basketballdrive\EncodedBy1920x1080';
% DecoderRoute='E:\matlab_code\ChecksumSimulations\ForH264\UpdateErrorGeneration\Decoder';
% 
% OutRoute='E:\2015,s10,summer\PercentageOfSynchBits\basketballdrive';
% 
% Type='Intact'; %'Intact' 'JM' 'STBMA' 'MldHard' 'FG'

%seqName = 'Crew';

%In265Route = ['F:\ICIP2020\Firouzeh\' seqName '\HEVC\encoding'];
%In265Name = [seqName '_qp32'];
%OutRoute = ['F:\ICIP2020\Firouzeh\' seqName '\HEVC\decoding\qp32'];
%MainRouteResult='F:\ICIP2020\Firouzeh'; % folder path
%DecoderRoute= [MainRouteResult '\MatlabCode\DecoderHEVC'];
%Type = 'Intact';

%% initialize decoder route
cd([DecoderRoute]);
%% initialize output name


%outText=['outF_' In264Name ];



%% copy 265 file into decoder folder
copyfile([In265Route APPEND_CHAR In265Name '.265'] ,[DecoderRoute APPEND_CHAR In265Name '.265' ]);

%% decode based on the type
switch Type
    case 'Intact'
        OutYuv=['dec_' In265Name '_Intact']; 
%         [d text]=system(['TAppDecoder_Release.exe -b ' In265Name '.265 -o ' OutYuv '.yuv']);
        system(['ffmpeg -ec guess_mvs+favor_inter+deblock -i ' In265Name '.265 ' OutYuv '.yuv']);
        d = 'd';
        text = 'text';
        
    case 'FFMPEG-FC'
        OutYuv=['dec_' In265Name '_FFMPEG-FC'];  
        system(['ffmpeg -ec guess_mvs+favor_inter+deblock -i ' In265Name '.265 ' OutYuv '.yuv 2> ffmpeg_FC.txt']);
        d = 'd';
        text = 'text';
        
    case 'HM-FC'
        OutYuv=['dec_' In265Name '_HM-FC'];  
        [d text]=system(['TAppDecoder_FC.exe -b ' In265Name '.265 -o ' OutYuv '.yuv']);
            
    case 'CRC_FFMPEG'
        OutYuv=['dec_' In265Name '_FFMPEG']; 
        system(['ffmpeg -ec guess_mvs+favor_inter+deblock -i ' In265Name '.265 ' OutYuv '.yuv 2> ffmpeg_CRC.txt']);
        d = 'd';
        text = 'text';
        
    case 'CRC_HM'
        OutYuv=['dec_' In265Name '_HM']; 
        [d text]=system(['TAppDecoder_Release.exe -b ' In265Name '.265 -o ' OutYuv '.yuv']);


end
%% copy output into output folder
copyfile([ DecoderRoute APPEND_CHAR In265Name '.265'],[OutRoute APPEND_CHAR In265Name '.265' ]);
if isfile([ DecoderRoute APPEND_CHAR OutYuv '.yuv'])
    copyfile([ DecoderRoute APPEND_CHAR OutYuv '.yuv'],[OutRoute APPEND_CHAR OutYuv '.yuv' ]);
    %% delete extra file in decoder folder
    delete([DecoderRoute APPEND_CHAR OutYuv '.yuv']);
end
delete([DecoderRoute APPEND_CHAR In265Name '.265']);
end

