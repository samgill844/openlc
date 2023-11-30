import pyopencl as _cl
import numpy as _np
from importlib_resources import files as _files
from openlc.binarystar.utils import transit_width
import matplotlib.pyplot as plt

def get_template_match_threshold(noise, cadence, radius_1, k, b, period, runtime, plot=True):
    incl = _np.arccos(radius_1*b)
    width = transit_width(radius_1,k, b, period=period)
    epochs = _np.linspace(-0.5*width, 0.5*width, 60)
    x = _np.arange(0, width, cadence).astype(_np.float64)
    ye = _np.random.uniform(0.5*noise, 1.5*noise, x.shape).astype(_np.float64)

    depth = 1 - lc(_np.array([0]), t_zero=0., ld_law=-2, period=period, radius_1=radius_1, k=k, incl=incl, runtime=runtime, ).astype(_np.float64)[0]


    total_maxs = []
    total_limits = []
    
    for _ in range(10):
        # Create the data to inject
        max_Ls = _np.empty_like(epochs)
        limit=-1
        for j in range(len(epochs)):
            model = lc(x, t_zero=epochs[j], ld_law=-2, period=period, radius_1=radius_1, k=k, incl=incl, runtime=runtime, ).astype(_np.float64)
            depth = 1 - lc(_np.array([epochs[j]]), t_zero=epochs[j], ld_law=-2, period=period, radius_1=radius_1, k=k, incl=incl, runtime=runtime)[0]
            normalisation_model = _np.ones(x.shape).astype(_np.float64)

            y = _np.random.normal(model , ye).astype(_np.float64)
            _, LL1 = template_match_lightcurve(x, y, ye, normalisation_model,runtime=runtime, offset=False, time_trial=epochs, period=period, radius_1=radius_1, k=k, incl=incl)
            max_Ls[j] = LL1.max()
            if (limit==-1) and ((1 - model.min()) > (0.9*depth)) : limit=LL1.max()
        total_limits.append(limit)
        total_maxs.append(max_Ls)
        
    if plot:
        fig, ax = plt.subplots(1,1)
        ax.plot(epochs, _np.median(total_maxs, axis=0) , c='orange')
        ax.fill_between(epochs, _np.median(total_maxs, axis=0) - _np.std(total_maxs, axis=0),
                        _np.median(total_maxs, axis=0) + _np.std(total_maxs, axis=0), color='orange', alpha = 0.3)
        ax.fill_between(epochs, _np.median(total_maxs, axis=0) - 2*_np.std(total_maxs, axis=0),
                        _np.median(total_maxs, axis=0) + 2*_np.std(total_maxs, axis=0), color='orange', alpha = 0.3)
        ax.fill_between(epochs, _np.median(total_maxs, axis=0) - 3*_np.std(total_maxs, axis=0),
                        _np.median(total_maxs, axis=0) + 3*_np.std(total_maxs, axis=0), color='orange', alpha = 0.3)
        ax.fill_between(epochs, _np.ones(epochs.shape)*(_np.median(total_limits) - _np.std(total_limits)),
                    _np.ones(epochs.shape)*(_np.median(total_limits) + _np.std(total_limits)), color='orange', alpha = 0.3)
        ax.fill_between(epochs, _np.ones(epochs.shape)*(_np.median(total_limits) - 2*_np.std(total_limits)),
                        _np.ones(epochs.shape)*(_np.median(total_limits) + 2*_np.std(total_limits)), color='orange', alpha = 0.3)
        ax.fill_between(epochs, _np.ones(epochs.shape)*(_np.median(total_limits) - 3*_np.std(total_limits)),
                        _np.ones(epochs.shape)*(_np.median(total_limits) + 3*_np.std(total_limits)), color='orange', alpha = 0.3)
        xticks = _np.linspace(-width/2,width/2,6)
        xtick_labels = ['{:.0f}\n{:.2f}'.format(int(1440*(i+width/2)), int(1440*(i+width/2))/60) for i in xticks]
        xtick_labels[0] = '0 mins\n0 hr'
        ax.set_xticks(xticks, xtick_labels)
        ax.set(xlabel='Time in transit', ylabel = r'$\Delta \log \mathcal{L}$', title='Suggest threshold {:.0f}\nDepth {:.1f} ppt [noise of {:.1f} ppt]'.format(_np.median(total_limits), 1e3*depth, 1e3*noise))
        plt.tight_layout()
        return limit, fig, ax
    else : return limit




def lc(t = _np.linspace(0,1,100), t_zero=0., period = 1.,
        radius_1=0.2, k = 0.2, incl=_np.pi/2,
        e=0., w = _np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=0, noversample=10,
        light_3=0.,
        ld_law = 2,
        accurate_tp=1,
		runtime=None, use_workspace=False):
	# Now set up the parameters
	t = t.astype(_np.float64)
	t_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=t)
	flux_g = _cl.Buffer(runtime.ctx, runtime.mf.WRITE_ONLY, t.nbytes)
	shape = t.shape

	# Get the shapes
	if use_workspace:
		work_group_size = runtime.max_work_group_size
		N_work_groups = int(_np.ceil(t.shape[0]/work_group_size))
		global_size = (N_work_groups*work_group_size,)
		work_group_size = (work_group_size,)		
	else : global_size, work_group_size = t.shape, None

	# Now call the kernel
	runtime.kernel_lc(runtime.queue, global_size, work_group_size, 
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
	_cl.enqueue_copy(runtime.queue, flux, flux_g)
	return flux


def reduce(t = _np.random.normal(1,0.1,int(1000)), runtime=None):
	# Check CPU usage
	if (runtime.max_work_group_size > t.shape[0]) or (runtime.max_work_group_size==1) : return t.sum()

	# Data
	t = t.astype(_np.float64)
	t_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=t)

	N_work_groups = _np.inf
	input_array_size = t.shape[0]
	work_group_size = (runtime.max_work_group_size,)
	i = 0
	while True:
		# Get the shapes
		N_work_groups = int(_np.ceil(input_array_size/work_group_size[0]))
		global_size = (N_work_groups*work_group_size[0],)

		# Define the output array
		result_g = _cl.Buffer(runtime.ctx, runtime.mf.WRITE_ONLY, size=8*N_work_groups)

		# Call
		runtime.kernel_reduce(runtime.queue, global_size, work_group_size, 
				t_g, result_g , _cl.LocalMemory(8*work_group_size[0]))
		
		if N_work_groups==1 : break 
		else:
			input_array_size = N_work_groups
			t_g = result_g 

	# Copy the work array back
	result = _np.zeros(N_work_groups, dtype=_np.float64)
	_cl.enqueue_copy(runtime.queue, dest=result, src=result_g, is_blocking=True)

	return result.sum()




def reduce_loglike(t = _np.random.normal(1,0.1,int(1000)), runtime=None):
	# Check CPU usage
	if (runtime.max_work_group_size > t.shape[0]) or (runtime.max_work_group_size==1) : return t.sum()

	# Data
	t = t.astype(_np.float64)
	t_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=t)

	N_work_groups = _np.inf
	input_array_size = t.shape[0]
	work_group_size = (runtime.max_work_group_size,)
	first = True
	while True:
		# Get the shapes
		N_work_groups = int(_np.ceil(input_array_size/work_group_size[0]))
		global_size = (N_work_groups*work_group_size[0],)

		# Define the output array
		result_g = _cl.Buffer(runtime.ctx, runtime.mf.WRITE_ONLY, size=8*N_work_groups)

		# Call
		if first:
			runtime.kernel_reduce_loglike(runtime.queue, global_size, work_group_size, 
					t_g, result_g , _cl.LocalMemory(8*work_group_size[0]))
			first=False
		else:
			runtime.kernel_reduce(runtime.queue, global_size, work_group_size, 
					t_g, result_g , _cl.LocalMemory(8*work_group_size[0]))	
		if N_work_groups==1 : break 
		else:
			input_array_size = N_work_groups
			t_g = result_g 

	# Copy the work array back
	result = _np.zeros(N_work_groups, dtype=_np.float64)
	_cl.enqueue_copy(runtime.queue, dest=result, src=result_g, is_blocking=True)

	return result.sum()






def lc_loglike(time, flux, flux_err, t_zero=0., period = 1.,
        radius_1=0.2, k = 0.2, incl=_np.pi/2,
        e=0., w = _np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=0, noversample=10,
        light_3=0.,
        ld_law = 2,
        accurate_tp=1,
		runtime=None, use_workspace=True,
		jitter=0., offset=1):
	# Check CPU usage
	if (runtime.max_work_group_size > time.shape[0]) or (runtime.max_work_group_size==1) : 
		model = lc(t = time, t_zero=t_zero, period = period,
				radius_1=radius_1, k = k, incl=incl,
				e=e, w = w,
				c = c, alpha = alpha,
				cadence=cadence, noversample=noversample,
				light_3=light_3,
				ld_law = ld_law,
				accurate_tp=accurate_tp,
				runtime=runtime, use_workspace=use_workspace)
		wt = 1/(flux_err**2 + jitter**2)
		loglikeliehood = -0.5*((flux - model)*(flux - model)*wt)
		return loglikeliehood.sum()

	# Data
	time, flux, flux_err = time.astype(_np.float64), flux.astype(_np.float64), flux_err.astype(_np.float64)
	time_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=time)
	flux_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=flux)
	flux_err_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=flux_err)

	N_work_groups = _np.inf
	input_array_size = time.shape[0]
	work_group_size = (runtime.max_work_group_size,)
	first = True
	while True:
		# Get the shapes
		N_work_groups = int(_np.ceil(input_array_size/work_group_size[0]))
		global_size = (N_work_groups*work_group_size[0],)

		# Define the output array
		result_g = _cl.Buffer(runtime.ctx, runtime.mf.WRITE_ONLY, size=8*N_work_groups)

		# Call
		if first:
			runtime.kernel_lc_loglike(runtime.queue, global_size, work_group_size, 
				time_g, flux_g, flux_err_g, result_g, _cl.LocalMemory(8*work_group_size[0]), _np.int32(time.shape[0]),
				_np.float64(t_zero), _np.float64(period),
				_np.float64(radius_1), _np.float64(k), _np.float64(incl),
				_np.float64(e), _np.float64(w), 
				_np.float64(c), _np.float64(alpha),
				_np.float64(cadence), _np.int32(noversample),
				_np.float64(light_3),
				_np.int32(ld_law),
				_np.int32(accurate_tp),
				_np.float64(jitter), _np.int32(offset))
			first=False
		else:
			runtime.kernel_reduce(runtime.queue, global_size, work_group_size, 
					t_g, result_g , _cl.LocalMemory(8*work_group_size[0]))	
		if N_work_groups==1 : break 
		else:
			input_array_size = N_work_groups
			t_g = result_g 

	# Copy the work array back
	result = _np.zeros(N_work_groups, dtype=_np.float64)
	_cl.enqueue_copy(runtime.queue, dest=result, src=result_g, is_blocking=True)

	return result.sum()



from openlc.binarystar.utils import transit_width
import numba
@numba.njit
def check_points(time, time_trial, time_trial_mask, width):
    for j in numba.prange(time_trial.shape[0]):
        for i in range(time.shape[0]):
            if abs(time[i] - time_trial[j]) < width : time_trial_mask[j] = True


def check_proximity_of_timestamps(time_trial, time, width, runtime,
				  use_workspace=True):
	# Data
	time_trial = time_trial.astype(_np.float64)
	time = time.astype(_np.float64)

	time_trial_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=time_trial)
	time_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=time)
	mask_g = _cl.Buffer(runtime.ctx, runtime.mf.WRITE_ONLY, size=4*time_trial.shape[0])

	# Get the shapes
	if use_workspace:
		work_group_size = runtime.max_work_group_size
		N_work_groups = int(_np.ceil(time_trial.shape[0]/work_group_size))
		global_size = (N_work_groups*work_group_size,)
		work_group_size = (work_group_size,)		
	else : global_size, work_group_size = time_trial.shape, None

	# Call
	runtime.check_proximity_of_timestamps(runtime.queue, global_size, work_group_size, 
		time_trial_g, time_g, _np.int32(time.shape[0]),_np.float64(width),
		mask_g)
		
	# Copy the work array back
	mask = _np.zeros(time_trial.shape[0], dtype=_np.int32)
	_cl.enqueue_copy(runtime.queue, dest=mask, src=mask_g, is_blocking=True)

	return mask.astype(bool)



def template_match_lightcurve(time, flux, flux_err, normalisation_model, period = 1.,
        radius_1=0.2, k = 0.2, incl=_np.pi/2,
        e=0., w = _np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=0, noversample=10,
        light_3=0.,
        ld_law = -2,
        accurate_tp=1,
		runtime=None, use_workspace=True,
		jitter=0., offset=0,
		time_step=None, time_trial=None):
	
	# Get the width
	width = transit_width(radius_1, k, _np.cos(incl)/radius_1, period=period)

	# Check the time steps
	if time_trial is None:
		if time_step is None : time_step = width / 20.
		time_trial = _np.arange(_np.min(time) - width/2., _np.max(time)+width/2., time_step)
		time_trial_mask = check_proximity_of_timestamps(time_trial, time, width, runtime, use_workspace=True)
		time_trial = time_trial[time_trial_mask]
	time_trial = time_trial.astype(_np.float64)

	# Data
	time, flux, flux_err = time.astype(_np.float64), flux.astype(_np.float64), flux_err.astype(_np.float64)
	time_trial_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=time_trial)
	DeltaL_g = _cl.Buffer(runtime.ctx, runtime.mf.WRITE_ONLY, size=8*time_trial.shape[0])

	time_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=time)
	flux_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=flux)
	flux_err_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=flux_err)
	normalisation_model_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=normalisation_model)


	# Get the shapes
	if use_workspace:
		work_group_size = runtime.max_work_group_size
		N_work_groups = int(_np.ceil(time_trial.shape[0]/work_group_size))
		global_size = (N_work_groups*work_group_size,)
		work_group_size = (work_group_size,)		
	else : global_size, work_group_size = t.shape, None

	# Call
	runtime.kernel_template_match_reduce(runtime.queue, global_size, work_group_size, 
		time_trial_g, DeltaL_g,
		time_g, flux_g, flux_err_g, normalisation_model_g, 
		_np.int32(time_trial.shape[0]), _np.int32(time.shape[0]),
		_np.float64(width),
		_np.float64(period),
		_np.float64(radius_1), _np.float64(k), _np.float64(incl),
		_np.float64(e), _np.float64(w), 
		_np.float64(c), _np.float64(alpha),
		_np.float64(cadence), _np.int32(noversample),
		_np.float64(light_3),
		_np.int32(ld_law),
		_np.int32(accurate_tp),
		_np.float64(jitter), _np.int32(offset))

	# Copy the work array back
	DeltaL = _np.zeros(time_trial.shape[0], dtype=_np.float64)
	_cl.enqueue_copy(runtime.queue, dest=DeltaL, src=DeltaL_g, is_blocking=True)

	return time_trial, DeltaL










def template_match_lightcurve_batch(time, flux, flux_err, normalisation_model, period = 1.,
        radius_1=_np.linspace(0.001,0.2,10), k = _np.linspace(0.001,0.2,10), b=_np.linspace(0.00,1,10),
        e=0., w = _np.pi/2.,
        c = 0.7, alpha = 0.4,
        cadence=0, noversample=10,
        light_3=0.,
        ld_law = -2,
        accurate_tp=1,
		runtime=None, use_workspace=True,
		jitter=0., offset=0,
		time_step=None, time_trial=None):
	
	# Check the time steps
	width = 1
	if time_trial is None:
		if time_step is None : time_step = width / 20.
		time_trial = _np.arange(_np.min(time) - width/2., _np.max(time)+width/2., time_step)
		time_trial_mask = check_proximity_of_timestamps(time_trial, time, width, runtime, use_workspace=True)
		time_trial = time_trial[time_trial_mask]
	time_trial = time_trial.astype(_np.float64)

	# Data
	time, flux, flux_err = time.astype(_np.float64), flux.astype(_np.float64), flux_err.astype(_np.float64)
	time_trial_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=time_trial)
	DeltaL_g = _cl.Buffer(runtime.ctx, runtime.mf.WRITE_ONLY, size=8*time_trial.shape[0]*radius_1.shape[0]*k.shape[0]*b.shape[0])
	radius_1_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=radius_1)
	k_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=k)
	b_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=b)

	time_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=time)
	flux_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=flux)
	flux_err_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=flux_err)
	normalisation_model_g = _cl.Buffer(runtime.ctx, runtime.mf.READ_ONLY | runtime.mf.COPY_HOST_PTR, hostbuf=normalisation_model)


	# Get the shapes
	if use_workspace:
		work_group_size = runtime.max_work_group_size
		N_work_groups = int(_np.ceil(time_trial.shape[0]/work_group_size))
		global_size = (N_work_groups*work_group_size,)
		work_group_size = (work_group_size,)		
	else : global_size, work_group_size = t.shape, None

	# Call
	runtime.kernel_template_match_batch_reduce(runtime.queue, global_size, work_group_size, 
		time_trial_g, DeltaL_g,
		time_g, flux_g, flux_err_g, normalisation_model_g, 
		_np.int32(time_trial.shape[0]), _np.int32(time.shape[0]),
		_np.float64(period),
		radius_1_g, k_g, b_g,
		_np.int32(radius_1.shape[0]),_np.int32(k.shape[0]),_np.int32(b.shape[0]),
		_np.float64(e), _np.float64(w), 
		_np.float64(c), _np.float64(alpha),
		_np.float64(cadence), _np.int32(noversample),
		_np.float64(light_3),
		_np.int32(ld_law),
		_np.int32(accurate_tp),
		_np.float64(jitter), _np.int32(offset))

	# Copy the work array back
	DeltaL = _np.zeros(time_trial.shape[0]*radius_1.shape[0]*k.shape[0]*b.shape[0], dtype=_np.float64)
	_cl.enqueue_copy(runtime.queue, dest=DeltaL, src=DeltaL_g, is_blocking=True)

	return time_trial, DeltaL.reshape((time_trial.shape[0],radius_1.shape[0],k.shape[0],b.shape[0]))