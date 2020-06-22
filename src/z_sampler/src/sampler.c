#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>


#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <math.h>

// Forward function declaration.
static PyObject *zsampler_sample_region(PyObject *self, PyObject *args);
static PyObject *zsampler_sample_categorical(PyObject *self, PyObject *args);


// Boilerplate: method list.
static PyMethodDef methods[] = {
  { "sample_region", zsampler_sample_region, METH_VARARGS, "Doc string."},
  { "sample_bulk_categorical", zsampler_sample_categorical, METH_VARARGS, "Doc string."},
  { NULL, NULL, 0, NULL } /* Sentinel */
};

static struct PyModuleDef sampler_module =
{
    PyModuleDef_HEAD_INIT,
    "zsampler", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC PyInit_zsampler(void)
{   
    import_array(); // crucial command if NumPy API is used
    return PyModule_Create(&sampler_module);
}

/*****************************************************************************
 * Helper methods                                                            *
 *****************************************************************************/
void print_int_array(npy_int64* array, int length) {
  for(int i = 0; i < length; i++) {
    printf("%ld ", array[i]);
  }
  printf("\n");
}

void print_f_array(npy_float64* array, int length) {
  for(int i = 0; i < length; i++) {
    printf("%lf ", array[i]);
  }
  printf("\n");
}

static PyObject* zsampler_sample_categorical(PyObject *self, PyObject *args) {


    PyArrayObject *py_Z, *py_prob;
    if (!PyArg_ParseTuple(args, "O!O!",
                            &PyArray_Type, &py_Z,
                            &PyArray_Type, &py_prob)) {
        return NULL;
    }

    npy_int64 C_N, C_K;
    C_N = PyArray_SHAPE(py_prob)[0];
    C_K = PyArray_SHAPE(py_prob)[1];

    const gsl_rng_type * C_T;
    gsl_rng * C_r;
    gsl_rng_env_setup();
    C_T = gsl_rng_taus;
    C_r = gsl_rng_alloc (C_T);

    double *_prob; // probabilities for sampling from multinomial
    _prob = (double*)malloc(C_K * sizeof(double));

    unsigned int *_new_sample;
    _new_sample = (unsigned int*)malloc(C_K * sizeof(unsigned int));

    PyArrayObject* out_Z = (PyArrayObject*)PyArray_NewCopy(py_Z, NPY_ANYORDER);
    for (int i=0; i < C_N; i++) {

        for (int k=0; k < C_K; k++) {
            _prob[k] = (*(npy_float64*)PyArray_GETPTR2(py_prob, i, k));
         }

        gsl_ran_multinomial(C_r, (size_t)C_K, 1, _prob, _new_sample);

        for (int k = 0; k < C_K; k++) {
            npy_int64* Zk = (npy_int64*)PyArray_GETPTR2(out_Z, i, k);
            Zk[0] = (npy_int64)_new_sample[k];
         }
    }

    free(_prob);
    free(_new_sample);
    gsl_rng_free(C_r);
    return (PyObject*)out_Z;  // no need to IncRef out_Z
}


//      Mixture weight priors is block-specific
static PyObject* zsampler_sample_region(PyObject *self, PyObject *args) {

  // Input variables
  PyArrayObject *py_Z, *py_counts, *py_covariates, *py_beta, *py_alpha;
  PyArrayObject *py_region_alloc, *py_region_label_counts;

  /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!",
                            &PyArray_Type, &py_Z,
                            &PyArray_Type, &py_counts,
                            &PyArray_Type, &py_covariates,
                            &PyArray_Type, &py_beta,
                            &PyArray_Type, &py_alpha,
                            &PyArray_Type, &py_region_alloc,
                            &PyArray_Type, &py_region_label_counts)) {
        return NULL;
    }

    Py_INCREF(py_region_alloc);
    Py_INCREF(py_region_label_counts);

    const gsl_rng_type * C_T;
    gsl_rng * C_r;
    gsl_rng_env_setup();
    C_T = gsl_rng_taus;
    C_r = gsl_rng_alloc (C_T);

    // Temporary variables
    npy_int64 C_N, C_K, C_J;
    C_N = PyArray_SHAPE(py_Z)[0];
    C_K = PyArray_SHAPE(py_Z)[1];
    C_J = PyArray_SHAPE(py_beta)[0];

    PyArrayObject* out_Z = (PyArrayObject*)PyArray_NewCopy(py_Z, NPY_ANYORDER);

    npy_int64 *C_counts = (npy_int64 *) PyArray_DATA(py_counts);
    npy_float64 *C_alpha = (npy_float64 *) PyArray_DATA(py_alpha);
    npy_int64 *C_region_alloc = (npy_int64 *) PyArray_DATA(py_region_alloc);

    npy_int64 C_labelsum;
    npy_int64 *C_label_counts;
    C_label_counts = (npy_int64*)malloc(C_K * sizeof(npy_int64));
    double *_mu;
    _mu = (double*)malloc(C_K * sizeof(double));
    double *_logpoi;
    _logpoi = (double*)malloc(C_K * sizeof(double));
    double *_logcat;
    _logcat =(double*)malloc(C_K * sizeof(double)); 
    double *_lik;
    _lik =(double*)malloc(C_K * sizeof(double)); 
    double *_prob; // probabilities for sampling from multinomial
    _prob = (double*)malloc(C_K * sizeof(double)); 
    unsigned int *_new_sample;
    _new_sample = (unsigned int*)malloc(C_K * sizeof(unsigned int));

    

    npy_float64 C_alpha_sum = 0.0;
    for(int k=0; k < C_K; k++) {
      C_alpha_sum = C_alpha_sum + C_alpha[k];
    }
    
    for (int i=0; i < C_N; i++) {

      C_labelsum = 0;
      for (int k = 0; k < C_K; k++) {
        // subtract current cell counts from the region counts
        npy_int64* accessor = (npy_int64*)PyArray_GETPTR2(py_region_label_counts, C_region_alloc[i], k);
        accessor[0] = accessor[0] - (*(npy_int64*)PyArray_GETPTR2(out_Z, i, k));  // TODO: make this simpler

        C_label_counts[k] = accessor[0];
        C_labelsum = C_labelsum + C_label_counts[k];
      }

      // computation of likelihood and interim calculations
      for (int k=0; k < C_K; k++) {
        _mu[k] = 0;
        for (int j=0; j < C_J; j++) {
            _mu[k] = _mu[k] + ((*(npy_float64*)PyArray_GETPTR2(py_covariates, i, j)) * (*(npy_float64*)PyArray_GETPTR2(py_beta, j, k)));
        }

        _logpoi[k] = ((double)C_counts[i])*_mu[k] - exp(_mu[k]);
        _logcat[k] = log(((double)C_label_counts[k] + C_alpha[k])  / ((double)C_labelsum + C_alpha_sum));

        _lik[k] = (_logpoi[k] + _logcat[k]);
      }

      double maxval = _lik[0];
      for (int k=0; k < C_K; k++) {
        if(_lik[k] > maxval) {
          maxval = _lik[k];
        }
      }

      double sumexp = 0;
      for (int k=0; k < C_K; k++) {
        sumexp = sumexp + exp((double)_lik[k] - maxval);
      }

      double logsumexp = maxval + log(sumexp);

      for (int k=0; k < C_K; k++) {
        _prob[k] = exp((double)_lik[k] - logsumexp);
      }

      gsl_ran_multinomial(C_r, (size_t)C_K, 1, _prob, _new_sample);
      for (int k = 0; k < C_K; k++) {
        npy_int64* Zk = (npy_int64*)PyArray_GETPTR2(out_Z, i, k);
        Zk[0] = (npy_int64)_new_sample[k];

        // update the region label counts
        npy_int64* accessor = (npy_int64*)PyArray_GETPTR2(py_region_label_counts, C_region_alloc[i], k);
        accessor[0] = accessor[0] + Zk[0];
      }
    }

    free(C_label_counts);
    free(_mu);
    free(_logpoi);
    free(_logcat);
    free(_lik);
    free(_prob);
    free(_new_sample);

    gsl_rng_free(C_r);
    Py_DECREF(py_region_alloc);
    Py_DECREF(py_region_label_counts);
    return (PyObject*)out_Z;  // no need to IncRef out_Z
}
