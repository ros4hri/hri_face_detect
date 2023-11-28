// Copyright (c) 2023 PAL Robotics S.L. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright (c) 2018-2021, Shiqi Yu, all rights reserved. shiqi.yu@gmail.com
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the {copyright_holder} nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Copied and adapted from https://github.com/ShiqiYu/libfacedetection

#include "facedetectcnn.h"


#if 0
#include <opencv2/opencv.hpp>
cv::TickMeter cvtm;
#define TIME_START cvtm.reset();cvtm.start();
#define TIME_END(FUNCNAME) cvtm.stop(); printf(FUNCNAME);printf("=%g\n", cvtm.getTimeMilli());
#else
#define TIME_START
#define TIME_END(FUNCNAME)
#endif


#define NUM_CONV_LAYER 53

extern ConvInfoStruct param_pConvInfo[NUM_CONV_LAYER];
Filters<float> g_pFilters[NUM_CONV_LAYER];

bool param_initialized = false;

void init_parameters()
{
    for(int i = 0; i < NUM_CONV_LAYER; i++)
        g_pFilters[i] = param_pConvInfo[i];
}

std::vector<FaceRect> objectdetect_cnn(unsigned char * rgbImageData, int width, int height, int step)
{

    TIME_START;
    if (!param_initialized)
    {
        init_parameters();
        param_initialized = true;
    }
    TIME_END("init");


    TIME_START;
    auto fx = setDataFrom3x3S2P1to1x1S1P0FromImage(rgbImageData, width, height, 3, step);
    TIME_END("convert data");

    /***************CONV0*********************/
    TIME_START;
    fx = convolution(fx, g_pFilters[0]);
    TIME_END("conv_head");

    TIME_START;
    fx = convolutionDP(fx, g_pFilters[1], g_pFilters[2]);
    TIME_END("conv0");

    TIME_START;
    fx = maxpooling2x2S2(fx);
    TIME_END("pool0");

    /***************CONV1*********************/
    TIME_START;
    fx = convolution4layerUnit(fx, g_pFilters[3], g_pFilters[4], g_pFilters[5], g_pFilters[6]);
    TIME_END("conv1");

    /***************CONV2*********************/
    TIME_START;
    fx = convolution4layerUnit(fx, g_pFilters[7], g_pFilters[8], g_pFilters[9], g_pFilters[10]);
    TIME_END("conv2");

    /***************CONV3*********************/
    TIME_START;
    fx = maxpooling2x2S2(fx);
    TIME_END("pool3");

    TIME_START;
    auto fb1 = convolution4layerUnit(fx, g_pFilters[11], g_pFilters[12], g_pFilters[13], g_pFilters[14]);
    TIME_END("conv3");

    /***************CONV4*********************/
    TIME_START;
    fx = maxpooling2x2S2(fb1);
    TIME_END("pool4");

    TIME_START;
    auto fb2 = convolution4layerUnit(fx, g_pFilters[15], g_pFilters[16], g_pFilters[17], g_pFilters[18]);
    TIME_END("conv4");

    /***************CONV5*********************/
    TIME_START;
    fx = maxpooling2x2S2(fb2);
    TIME_END("pool5");

    TIME_START;
    auto fb3 = convolution4layerUnit(fx, g_pFilters[19], g_pFilters[20], g_pFilters[21], g_pFilters[22]);
    TIME_END("conv5");

    CDataBlob<float> pred_reg[3], pred_cls[3], pred_kps[3], pred_obj[3];
    /***************branch5*********************/
    TIME_START;
    fb3 = convolutionDP(fb3, g_pFilters[27], g_pFilters[28]);
    pred_cls[2] = convolutionDP(fb3, g_pFilters[33], g_pFilters[34], false);
    pred_reg[2] = convolutionDP(fb3, g_pFilters[39], g_pFilters[40], false);
    pred_kps[2] = convolutionDP(fb3, g_pFilters[51], g_pFilters[52], false);
    pred_obj[2] = convolutionDP(fb3, g_pFilters[45], g_pFilters[46], false);
    TIME_END("branch5");

    /*****************add5*********************/    
    TIME_START;
    fb2 = elementAdd(upsampleX2(fb3), fb2);
    TIME_END("add5");

    /*****************add6*********************/    
    TIME_START;
    fb2 = convolutionDP(fb2, g_pFilters[25], g_pFilters[26]);
    pred_cls[1] = convolutionDP(fb2, g_pFilters[31], g_pFilters[32], false);
    pred_reg[1] = convolutionDP(fb2, g_pFilters[37], g_pFilters[38], false);
    pred_kps[1] = convolutionDP(fb2, g_pFilters[49], g_pFilters[50], false);
    pred_obj[1] = convolutionDP(fb2, g_pFilters[43], g_pFilters[44], false);
    TIME_END("branch4");

    /*****************add4*********************/
    TIME_START;
    fb1 = elementAdd(upsampleX2(fb2), fb1);
    TIME_END("add4");

    /***************branch3*********************/
    TIME_START;
    fb1 = convolutionDP(fb1, g_pFilters[23], g_pFilters[24]);
    pred_cls[0] = convolutionDP(fb1, g_pFilters[29], g_pFilters[30], false);
    pred_reg[0] = convolutionDP(fb1, g_pFilters[35], g_pFilters[36], false);
    pred_kps[0] = convolutionDP(fb1, g_pFilters[47], g_pFilters[48], false);
    pred_obj[0] = convolutionDP(fb1, g_pFilters[41], g_pFilters[42], false);
    TIME_END("branch3");
    
    /***************PRIORBOX*********************/
    TIME_START;
    auto prior3 = meshgrid(fb1.cols, fb1.rows, 8);
    auto prior4 = meshgrid(fb2.cols, fb2.rows, 16);
    auto prior5 = meshgrid(fb3.cols, fb3.rows, 32);
    TIME_END("prior");
    /***************PRIORBOX*********************/

    TIME_START;
    bbox_decode(pred_reg[0], prior3, 8);
    bbox_decode(pred_reg[1], prior4, 16);
    bbox_decode(pred_reg[2], prior5, 32);

    kps_decode(pred_kps[0], prior3, 8);
    kps_decode(pred_kps[1], prior4, 16);
    kps_decode(pred_kps[2], prior5, 32);

    auto cls = concat3(blob2vector(pred_cls[0]), blob2vector(pred_cls[1]), blob2vector(pred_cls[2]));
    auto reg = concat3(blob2vector(pred_reg[0]), blob2vector(pred_reg[1]), blob2vector(pred_reg[2]));
    auto kps = concat3(blob2vector(pred_kps[0]), blob2vector(pred_kps[1]), blob2vector(pred_kps[2]));
    auto obj = concat3(blob2vector(pred_obj[0]), blob2vector(pred_obj[1]), blob2vector(pred_obj[2]));

    sigmoid(cls);
    sigmoid(obj);
    TIME_END("decode")

    TIME_START;
    std::vector<FaceRect> facesInfo = detection_output(cls, reg, kps, obj, 0.45f, 0.5f, 1000, 512);
    TIME_END("detection output")
    return facesInfo;
}

int* facedetect_cnn(unsigned char * result_buffer, //buffer memory for storing face detection results, !!its size must be 0x9000 Bytes!!
    unsigned char * rgb_image_data, int width, int height, int step) //input image, it must be BGR (three-channel) image!
{

    if (!result_buffer)
    {
        fprintf(stderr, "%s: null buffer memory.\n", __FUNCTION__);
        return NULL;
    }
    //clear memory
    result_buffer[0] = 0;
    result_buffer[1] = 0;
    result_buffer[2] = 0;
    result_buffer[3] = 0;

    std::vector<FaceRect> faces = objectdetect_cnn(rgb_image_data, width, height, step);

    int num_faces =(int)faces.size();
    num_faces = MIN(num_faces, 1024); //1024 = 0x9000 / (16 * 2 + 4)

    int * pCount = (int *)result_buffer;
    pCount[0] = num_faces;

    for (int i = 0; i < num_faces; i++)
    {
        //copy data
        short * p = ((short*)(result_buffer + 4)) + 16 * size_t(i);
        p[0] = (short)(faces[i].score * 100);
        p[1] = (short)faces[i].x;
        p[2] = (short)faces[i].y;
        p[3] = (short)faces[i].w;
        p[4] = (short)faces[i].h;
        //copy landmarks
        for (int lmidx = 0; lmidx < 10; lmidx++)
        {
            p[5 + lmidx] = (short)faces[i].lm[lmidx];
        }
    }

    return pCount;
}
