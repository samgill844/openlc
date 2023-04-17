import pyopencl as _cl
import numpy as _np

def rv1(t = _np.linspace(0,1,100), 
        t_zero=0., period = 1.,
        K1=100,
        e=0., w = _np.pi/2.,incl=_np.pi/2,
        V0=0,
        accurate_tp=1,
		runtime=None):
	# Now set up the parameters
	t = t.astype(_np.float64)
	t_g = _cl.Buffer(runtime['ctx'], runtime['mf'].READ_ONLY | runtime['mf'].COPY_HOST_PTR, hostbuf=t)
	rv_g = _cl.Buffer(runtime['ctx'], runtime['mf'].WRITE_ONLY, t.nbytes)

	# Now call the kernel
	runtime['kernel_rv1'](runtime['queue'], t.shape, None, 
		t_g, rv_g, 
		_np.float64(t_zero), _np.float64(period),
		_np.float64(K1),
		_np.float64(e), _np.float64(w), _np.float64(incl),
		_np.float64(V0),
		_np.int32(accurate_tp))

	rv_data = _np.empty_like(t)
	_cl.enqueue_copy(runtime['queue'], rv_data, rv_g)
	return rv_data



def rv2(t = _np.linspace(0,1,100), 
        t_zero=0., period = 1.,
        K1=100,K2=20,
        e=0., w = _np.pi/2.,incl=_np.pi/2,
        V0=0,
        accurate_tp=1,
		runtime=None):
	# Now set up the parameters
	t = t.astype(_np.float64)
	t_g = _cl.Buffer(runtime['ctx'], runtime['mf'].READ_ONLY | runtime['mf'].COPY_HOST_PTR, hostbuf=t)
	rv_g1 = _cl.Buffer(runtime['ctx'], runtime['mf'].WRITE_ONLY, t.nbytes)
	rv_g2 = _cl.Buffer(runtime['ctx'], runtime['mf'].WRITE_ONLY, t.nbytes)

	# Now call the kernel
	runtime['kernel_rv2'](runtime['queue'], t.shape, None, 
		t_g, rv_g1, rv_g2, 
		_np.float64(t_zero), _np.float64(period),
		_np.float64(K1),_np.float64(K2),
		_np.float64(e), _np.float64(w), _np.float64(incl),
		_np.float64(V0),
		_np.int32(accurate_tp))

	rv_data_1, rv_data_2 = _np.empty_like(t), _np.empty_like(t)
	_cl.enqueue_copy(runtime['queue'], rv_data_1, rv_g1)
	_cl.enqueue_copy(runtime['queue'], rv_data_2, rv_g2)

	return rv_data_1, rv_data_2