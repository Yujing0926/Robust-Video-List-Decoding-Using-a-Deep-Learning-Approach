function [] = CandidatesGeneration(video_name,Qp,damage_rate,PacketNum,CandidatesRoute,DecoderRoute,ImageWidth,ImageHeight,yuvformat,CodeRoute,In265Route,APPEND_CHAR)
   
    SEQUENCE_NAME = video_name;
    %CandidatesFolder = append(CandidatesRoute,APPEND_CHAR,SEQUENCE_NAME,APPEND_CHAR,'qp',string(Qp));
    CandidatesFolder = append(CandidatesRoute);

    %% make folder for work
    if (~exist(CandidatesRoute))
        mkdir(CandidatesRoute)
    end

    if (~exist(CandidatesFolder))
        mkdir(CandidatesFolder)
    end

    inputyuv=append(In265Route,APPEND_CHAR,SEQUENCE_NAME,'_qp',string(Qp),'.yuv');

    ImgNum = PacketNum-3;
    ImgName = append(SEQUENCE_NAME,'_qp',string(Qp));
    cd(CodeRoute);
    save_yuvframe(In265Route,In265Route,ImgName,ImgNum,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR);

    inputfile1=append(In265Route,APPEND_CHAR,SEQUENCE_NAME,'_qp',string(Qp),'.265');
    disp(inputfile1);
    
    FileIDScan = fopen(inputfile1,'r');
    scanTab = scanHEVC(FileIDScan);
    TotalPacket = size(scanTab,2);
    scanTab = scanTab(1:TotalPacket); 
    fclose(FileIDScan);
    
    if ((sum(abs(scanTab - round(scanTab))))~=0)
        disp('******* ERROR *******')
        disp('******* ERROR *******')
        disp('******* ERROR *******')
        disp(video_name)
        scanTab
    end

    InputData = fopen(inputfile1,'r');
    pBit1 = fread(InputData,scanTab(1)*8,'ubit1','ieee-be');
    pBit2 = fread(InputData,scanTab(2)*8,'ubit1','ieee-be');
    pBit3 = fread(InputData,scanTab(3)*8,'ubit1','ieee-be');
    pBitHeader = {pBit1, pBit2, pBit3};
    
    pos_Error = int64(scanTab(PacketNum)*8*damage_rate);

    for i = pos_Error : pos_Error
        OutputFile = append(CandidatesFolder,APPEND_CHAR,SEQUENCE_NAME,"qp",string(Qp),"_",string(i),"_",string(damage_rate),".265");
        edit(OutputFile);
        OutputData = fopen(OutputFile,'w');
        for j = 1 : TotalPacket
            if j <= 3
                h265write(OutputData,cell2mat(pBitHeader(j)));
            else
                pBit=fread(InputData,scanTab(j)*8,'ubit1','ieee-be'); 
                if j == PacketNum
                    pBitchanged=pBit; 
                    pBitchanged(i) = abs(pBitchanged(i)+(-1));
                    h265write(OutputData,pBitchanged);
                else
                    h265write(OutputData,pBit);
                end
            end
        end
        fclose(OutputData);
        OutYuv = append(CandidatesFolder,APPEND_CHAR,SEQUENCE_NAME,"_qp",string(Qp),"_frame_",string(ImgNum),"_damaged_",string(damage_rate),".yuv");
        cd(DecoderRoute);
        system(append('ffmpeg -ec 0 -i ',OutputFile,' ',OutYuv));       
    end
    fclose(InputData);
    fclose('all');
    if exist(OutYuv, 'file') == 2
        OutImg = append(SEQUENCE_NAME,"_qp",string(Qp),"_frame_",string(ImgNum),"_damaged_",string(damage_rate));
        cd(CodeRoute);
        % save the RGB image
        save_yuvframe(CandidatesFolder,CandidatesFolder,OutImg,ImgNum,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR);
        % %  save the YUV image
        % save_yuv(CandidatesFolder,CandidatesFolder,OutImg,ImgNum,ImageWidth,ImageHeight,yuvformat,APPEND_CHAR);
    end
end
