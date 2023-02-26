//
// Created by Pierluigi Carlino on 18/02/23.
//

#include "imageHandler.h"
uint8_t distance(pixel a,pixel b){
    return (uint8_t) sqrt(
            pow(a.a-b.a,2)+
            pow(a.b-b.b,2)+
            pow(a.c-a.c,2)
    );
}
pixel getPixel(uint8_t *data){
    struct pixel toRet;
    toRet.a = data[0];
    toRet.b = data[1];
    toRet.c = data[2];

    return toRet;
}
pixel meanPixel(pixel a,pixel b,uint32_t prev){
    struct pixel toRet;
    prev +=1;

    toRet.a = (a.a*(prev-1)+b.a)/prev;
    toRet.b = (a.b*(prev-1)+b.b)/prev;
    toRet.c = (a.c*(prev-1)+b.c)/prev;

    return toRet;
}
void setPixel(uint8_t *data,pixel p){
    data[0] = p.a;
    data[1] = p.b;
    data[2] = p.c;
}

void initPixel(uint8_t *data){
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
}

void clusterData(uint8_t *src,uint8_t *cluster_points,uint8_t *out,uint32_t src_l,uint32_t cluster_l,uint32_t _distance){
    for (int  i = 0; i < src_l; i=i+3){
        pixel p_src = getPixel(src+i);
        pixel p_mean = getPixel(src+i);
        char found = 0;
        uint32_t  previouses = 0;

        #pragma omp parallel for
        for (int j = 0; j < cluster_l; j=j+3){
            pixel p_cls = getPixel(cluster_points+j);

            if(distance(p_cls,p_src)<_distance){
                setPixel(out+i,p_src);
                found = 1;
                break;
            }
        }

        if(!found)
            initPixel(out+i);

        #pragma omp parallel for
        //check if near is similar
        for (int  k = i+3; k < src_l; k=k+3){
            pixel p_src_2 = getPixel(src+k);


            if(distance(p_mean,p_src_2)<_distance){
                if(found)
                    setPixel(out+k,p_src_2);
                else
                    initPixel(out+k);

                p_mean = meanPixel(p_mean,p_src_2,previouses);
                i = k;
            }else{
                break;
            }
        }
    }

}



//zone dedicated for gray scale images
void removeThreshold(uint8_t *data,uint32_t w,uint32_t h,uint32_t threashold){
    //todo implementation
}


void removeBlockNoise(uint8_t *data,kernel dimensions,kernel kern_dim,uint32_t thr){
    //calculate the number of blocks
    kernel blocks;
    blocks.kern_w = ceil((float)dimensions.kern_w/(float)kern_dim.kern_w);
    blocks.kern_h = ceil((float)dimensions.kern_h/(float)kern_dim.kern_h);

    kernel  smallKernel;
    smallKernel.kern_w = dimensions.kern_w%kern_dim.kern_w;
    smallKernel.kern_h = dimensions.kern_h%kern_dim.kern_h;

    uint32_t  blocks_number = blocks.kern_w*blocks.kern_w;




    #pragma omp parallel for
    for(int b=0;b<blocks_number;b++){
        kernel k = getKernelDim(b,blocks,kern_dim,smallKernel);
        uint32_t  sum = 0;

        int i_off = (b%blocks.kern_w)*kern_dim.kern_w;

        //calculate avg
        #pragma omp parallel for

        for(int i=0;i<k.kern_w;i++){
            int j_off = (floor(b/blocks.kern_w))*kern_dim.kern_h;

            for(int j = 0;j<k.kern_h;j++) {
                uint32_t  off = i+i_off+dimensions.kern_w*(j+j_off);
                sum += data[off];
            }
        }

        sum = floor(sum/(k.kern_w*k.kern_h));


        //remove outliers
        uint8_t  toSet = thr>sum ? 0x0 : 0xff;

        #pragma omp parallel for
        for(int i=0;i<k.kern_w;i++){
            int j_off = (floor(b/blocks.kern_w))*kern_dim.kern_h;

            for(int j = 0;j<k.kern_h;j++) {
                uint32_t  off = i+i_off+dimensions.kern_w*(j+j_off);
                data[off] = toSet;
            }
        }

    }

}


block_type getBlockType(uint32_t block,kernel blocks){
    block +=1;



    if( block>blocks.kern_w && (uint32_t )
                                       (1+floor(block/blocks.kern_w))%blocks.kern_h == 0)
        return BOTTOM_BLOCK;
    if(block%blocks.kern_w == 0 &&  block>blocks.kern_w &&  (uint32_t)(floor(block/blocks.kern_w))%blocks.kern_h == 0)
        return LEFT_BOTTOM_BLOCK;

    if(block%blocks.kern_w == 0)
        return LEFT_BLOCK;
    return NORMAL_BLOCK;

}
kernel getKernelDim(uint32_t block,kernel blocks,kernel bigKernel,kernel smallKernel){
    block_type  blockType = getBlockType(block,blocks);
    if(blockType == NORMAL_BLOCK)
        return bigKernel;

    kernel toRet;

    if(blockType == LEFT_BLOCK){
        toRet.kern_h = bigKernel.kern_h;
        toRet.kern_w = smallKernel.kern_w;

    }else if(blockType == BOTTOM_BLOCK){
        toRet.kern_h = smallKernel.kern_h;
        toRet.kern_w = bigKernel.kern_w;

    }else{
        return smallKernel;
    }

    return toRet;
}


fourierGrid* gridData(double *dataToGrid,uint32_t dataLen,uint8_t precision){
    fourierGrid *dataGrid = NULL;

    for(uint32_t i=0;i<dataLen*2;i+=2) {
        double x = getPrecision(precision, dataToGrid[i]);
        double y = getPrecision(precision, dataToGrid[i + 1]);


        //todo the founded pointer can be used to make
        //a sort of hash table to reduce the time complexity
        //for the next searches
        fourierGrid *founded = findGrid(dataGrid, x, y);


        if (dataGrid == NULL)
            dataGrid = founded;

        for (uint32_t j = i + 2; i < dataLen * 2; j += 2, i += 2) {
            double x_next = getPrecision(precision, dataToGrid[j]);
            double y_next = getPrecision(precision, dataToGrid[j + 1]);

            if (x_next == x && y_next == y)
                continue;
            else
                break;
        }
    }

    //prepare double to ret then free the list
    return dataGrid;
}

double *convertGridToArray(fourierGrid *dataGrid){
    double* toRet = malloc(sizeof(double )*dataGrid->len*2);
    uint32_t j=0;
    fourierGrid *iterator = dataGrid;
    while(iterator!=NULL){
        fourierGrid *freeIter = iterator;
        iterator = iterator->next;

        toRet[j++] = freeIter->x;
        toRet[j++] = freeIter->y;

        free(freeIter);
    };

    return toRet;
}
void freeFourier(fourierGrid *toFree){
    fourierGrid *iterator = toFree;
    while(iterator!=NULL){
        fourierGrid *freeIter = iterator;
        iterator = iterator->next;
        free(freeIter);
    };
}

double getPrecision(uint8_t precision,double  toGet){
    double  factor = pow(10,precision);
    return  floor(factor*toGet)/factor;
}

fourierGrid *findGrid(struct fourierGrid *whereToFind,double x,double y){

    fourierGrid *toRet = NULL;
    if(whereToFind == NULL){
        toRet = malloc(sizeof(struct fourierGrid));
        toRet->x = x;
        toRet->y = y;
        toRet->next = NULL;
        toRet->len = 1;
    }else{
        fourierGrid *iterator = whereToFind;
        while(iterator->next!=NULL){
            if(iterator->x == x && iterator->y == y) {
                toRet = iterator;
                break;
            }
            iterator = iterator->next;
        };

        if(toRet == NULL ) {

            if (iterator->x == x && iterator->y == y) {
                toRet = iterator;

            } else {
                toRet = malloc(sizeof(struct fourierGrid));
                toRet->x = x;
                toRet->y = y;
                toRet->next = NULL;
                iterator->next = toRet;
                whereToFind->len += 1;

            }
        }
    }

    return toRet;
}


double *convertToSampledFourier(double *fourierCoff,uint32_t granularity,uint32_t fourier_points){
    uint64_t  desired_N = pow((2*pow(10,granularity)),2);
    uint32_t basicStep = floor(desired_N/fourier_points);

    double *new_coefficients = malloc(sizeof (double )*desired_N*4);
    memset(new_coefficients,0,sizeof (double )*desired_N*4);

    new_coefficients[0] = fourierCoff[0];
    new_coefficients[1] = fourierCoff[1];
    new_coefficients[2] = fourierCoff[2];
    new_coefficients[3] = fourierCoff[3];

    for(uint32_t  i=4;i<4*fourier_points;i+=4){
        uint32_t  off = i*basicStep%desired_N;
        new_coefficients[off] = fourierCoff[i];
        new_coefficients[off+1] = fourierCoff[i+1];
        new_coefficients[off+2] = fourierCoff[i+2];
        new_coefficients[off+3] = fourierCoff[i+3];
    }

    return new_coefficients;


}