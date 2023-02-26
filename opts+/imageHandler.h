//
// Created by Pierluigi Carlino on 18/02/23.
//

#ifndef TEST_IMAGEHANDLER_H
#define TEST_IMAGEHANDLER_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <printf.h>
#include <string.h>

typedef struct pixel{
    uint8_t a;
    uint8_t b;
    uint8_t c;
}pixel;



#define BOTTOM_BLOCK 0
#define LEFT_BLOCK 1
#define LEFT_BOTTOM_BLOCK 2
#define NORMAL_BLOCK 3

typedef struct kernel{
    uint32_t  kern_w;
    uint32_t  kern_h;
}kernel;

typedef uint8_t  block_type;



uint8_t distance(pixel a,pixel b);
pixel getPixel(uint8_t *data);
pixel meanPixel(pixel a,pixel b,uint32_t prev);

void setPixel(uint8_t *data,pixel p);
void initPixel(uint8_t *data);
void clusterData(uint8_t *src,uint8_t *cluster_points,uint8_t *out,uint32_t src_l,uint32_t cluster_l,uint32_t _distance);


//zone dedicated for gray scale images
void removeThreshold(uint8_t *data,uint32_t w,uint32_t h,uint32_t threashold);
void removeBlockNoise(uint8_t *data,kernel dimensions,kernel kern_dim,uint32_t thr);


block_type getBlockType(uint32_t block,kernel blocks);
kernel getKernelDim(uint32_t block,kernel blocks,kernel bigKernel,kernel smallKernel);

//zone dedicated for fourier preprocessing
typedef struct fourierGrid{
    double x;
    double y;
    uint32_t  len;
    struct fourierGrid *next;
}fourierGrid;

fourierGrid* gridData(double *dataToGrid,uint32_t dataLen,uint8_t precision);
fourierGrid *findGrid(struct fourierGrid *whereToFind,double x,double y);
double getPrecision(uint8_t precision,double  toGet);
void freeFourier(fourierGrid *toFree);
double *convertGridToArray(fourierGrid *dataGrid);

double *convertToSampledFourier(double *fourierCoff,uint32_t granularity,uint32_t fourier_points);

#endif //TEST_IMAGEHANDLER_H
