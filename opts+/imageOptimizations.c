#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdint.h>
#include "imageHandler.c"



static PyObject* filterWithArray(PyObject* self, PyObject* args)
{
    PyArrayObject *matrix1, *matrix2, *result;
    int _distance;

    int ndim;
    npy_intp *shape;
    uint8_t *data_ptr, *data_ptr2, *out_ptr;
    int n,n2;

    if (!PyArg_ParseTuple(args, "OOi", &matrix1, &matrix2,&_distance))
        return NULL;


    n = PyArray_SIZE(matrix1);
    n2 = PyArray_SIZE(matrix2);

    if (n % 3 != 0 || n2 % 3 != 0) {
        PyErr_SetString(PyExc_ValueError, "Input  must be multiple of 3");
        return NULL;
    }



    ndim = PyArray_NDIM(matrix1);
    shape = PyArray_DIMS(matrix1);

    result = (PyArrayObject *) PyArray_SimpleNew(ndim, shape, NPY_UINT8);
    if (result == NULL)
        return NULL;

    data_ptr = (uint8_t *) PyArray_DATA(matrix1);
    data_ptr2 = (uint8_t *) PyArray_DATA(matrix2);
    out_ptr = (uint8_t *) PyArray_DATA(result);


    clusterData(data_ptr,data_ptr2,out_ptr,n,n2,_distance);

    return PyArray_Return(result);
}
static PyObject* removeNoiseGrayScale(PyObject* self, PyObject* args)
{
    PyArrayObject *matrix;
    uint32_t  kern_w,kern_h,thr;

    uint8_t *data_ptr;
    if (!PyArg_ParseTuple(args, "Oiii", &matrix, &kern_w,&kern_h,&thr))
        return NULL;

    data_ptr = (uint8_t *) PyArray_DATA(matrix);
    npy_intp* shape = PyArray_SHAPE(matrix);

    kernel  dimensions;
    dimensions.kern_w = shape[1];
    dimensions.kern_h = shape[0];

    kernel  kern_dim;
    kern_dim.kern_w =kern_w;
    kern_dim.kern_h = kern_h;

    removeBlockNoise(data_ptr,dimensions,kern_dim,thr);

    return PyArray_Return(matrix);
}
static PyObject* prepareFourierGrid(PyObject* self, PyObject* args)
{
    PyArrayObject *matrix;
    uint32_t granularity;

    double *data_ptr;

    if (!PyArg_ParseTuple(args, "Oi", &matrix, &granularity))
        return NULL;

    data_ptr = (double *) PyArray_DATA(matrix);
    npy_intp* shape = PyArray_SHAPE(matrix);

    fourierGrid *grid = gridData(data_ptr,shape[0],granularity);
    uint32_t  returned = grid->len;
    double *newData = convertGridToArray(grid);

    // Create a new NumPy array with the given data pointer
    npy_intp dims[] = {returned, 2}; // Array dimensions
    PyObject* array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, newData);


    return PyArray_Return(array);
}

static PyObject* fixedFourier(PyObject* self, PyObject* args)
{
    PyArrayObject *matrix;
    uint32_t granularity;

    double *data_ptr;

    if (!PyArg_ParseTuple(args, "Oi", &matrix, &granularity))
        return NULL;

    double *coeff  = (double *) PyArray_DATA(matrix);
    npy_intp* shape = PyArray_SHAPE(matrix);

    uint32_t returned = shape[0];

    double *new_coff = convertToSampledFourier(coeff,granularity,returned);

    uint64_t  desired_N = pow((2*pow(10,granularity)),2);

    npy_intp toRetdims[] = {desired_N, 1}; // Array dimensions
    PyObject* new_coffs = PyArray_SimpleNewFromData(2, toRetdims, NPY_CDOUBLE, new_coff);

    return PyArray_Return(new_coffs);
}


static PyMethodDef module_methods[] = {
        {"filterWithArray", filterWithArray, METH_VARARGS, "Filter an array a using array b and a distance factor"},
        {"remove_noise_gs", removeNoiseGrayScale, METH_VARARGS, "Remove noise of a image using a block dimension"},
        {"prepareFourierGrid", prepareFourierGrid, METH_VARARGS, "Given an array of integer and a granularity prepare a fourier grid"},
        {"fixedFourier", prepareFourierGrid, METH_VARARGS, "Given an array of integer and a granularity prepare a fourier grid"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "scan",
    "Example module",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_scan(void)
{
    PyObject *m;
    import_array();
    m = PyModule_Create(&module_def);
    if (m == NULL)
        return NULL;
    return m;
}
