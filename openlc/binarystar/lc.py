import pyopencl as _cl
import numpy as _np

def lc(t = _np.linspace(0,1,100), t_zero=0., period = 1.,
        radius_1=0.2, k = 0.2, incl=_np.pi/2,
        e=0., w = _np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=0, noversample=10,
        light_3=0.,
        ld_law = 2,
        accurate_tp=1,
		runtime=None):
	# Now set up the parameters
	t = t.astype(_np.float64)
	t_g = _cl.Buffer(runtime['ctx'], runtime['mf'].READ_ONLY | runtime['mf'].COPY_HOST_PTR, hostbuf=t)
	flux_g = _cl.Buffer(runtime['ctx'], runtime['mf'].WRITE_ONLY, t.nbytes)

	# Now call the kernel
	runtime['kernel_lc'](runtime['queue'], t.shape, None, 
		t_g, flux_g, 
		_np.float64(t_zero), _np.float64(period),
		_np.float64(radius_1), _np.float64(k), _np.float64(incl),
		_np.float64(e), _np.float64(w), 
		_np.float64(c), _np.float64(alpha),
		_np.float64(cadence), _np.int32(noversample),
		_np.float64(light_3),
		_np.int32(ld_law),
		_np.int32(accurate_tp))

	flux = _np.empty_like(t)
	_cl.enqueue_copy(runtime['queue'], flux, flux_g)
	return flux

def reduce(t = _np.arange(100), runtime=None):

	# Get device and context, create command queue and program
	context = runtime['ctx']
	queue = runtime['queue']

	# Data
	data = _np.empty(shape=(2,), dtype=_np.int32)

	# Create input/output buffer
	data_buff = _cl.Buffer(context, runtime['mf'].WRITE_ONLY, size=data.nbytes)

	# Enqueue kernel
	global_size = (100,)
	local_size = (4,)

	# __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)
	runtime['kernel_reduce'](queue, global_size, local_size, data_buff)

	# Print averaged results
	_cl.enqueue_copy(queue, dest=data, src=data_buff, is_blocking=True)

	print('Increment: ' + str(data[0]))
	print('Atomic increment: ' + str(data[1]))