For the original videos downloaded from Internet ([https://media.xiph.org/video/derf/](https://media.xiph.org/video/derf/), [https://mcl.usc.edu/mcl-jcv-dataset/](https://mcl.usc.edu/mcl-v-database/) and [https://www.cdvl.org/](https://www.cdvl.org/)) :

You can see the file **90 original video sequences used.txt** to find the original video names from different sites.

1. Use _Cut_video_main.m_ to generate original YUV videos in resolution 1920x1024.
   - **MainResultRoute** : your main path of all results;
   - **AviRoute** : your path to save the original videos from the Internet;
   - **test_rate** : choose the percentage between [0,1] to randomly split the video sequences for training and inference.
2. Use _Candidate_Generation_rgb.m_ to generate candidate videos and extract image patch database for training or test.
   - In this file, you can run the codes to generate candidates with 1 bit flipped. You should change the following parameters :
      - **SYSTEM** : your system of your computer to run this code;
      - **MainResultRoute** : your main path of all the codes and results (others are the relative paths); 
      - **PacketNum** : choose which packet you want damaged (normally 4 is the intra frame packet and 5 is the first inter frame packet);
      - **frame** : choose the frame type (intra or inter);
      - **damage_rate_list** : define the position of the flipped bit in the packet (bit position = damage_rate x total_bit_in_this_packet);
      - **QPlist** : the QP of the encoded videos.
      - **patch_size** : choose the generated patch size (ex: 32, 64, 128, etc.);
      - **DCTT** : use the Discriminant Color Texture Transformation color space conversion (0: use, 1: not use, by default is 0).
   - Note that to encode the original video sequence, you need to download the file **encoder_lowdelay_main.cfg** in the main directory of this Git in order to encode the videos successfully. And we use the ffmpeg to decode the encoded videos, download the proper ffmpeg version for your system [here](https://www.ffmpeg.org/download.html) and put it in this folder and the folder **DecoderHEVC**.
3. Use _extract_database_rgb.py_ to make the .txt and .xlsx files including reference information of the extracted image patch database.
   - **main_route** : your main path of all the codes and results (others are the relative paths);
   - **function** : choose the data type (train or test);
   - **error_frame**: choose the frame type (1 for intra or 2 for inter).
