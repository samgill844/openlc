import pyopencl as _cl 
from importlib_resources import files as _files


__all__ = ['create_context', 'create_queue', 'create_context_and_queue']

def create_context(answers=[0,0]):
    return _cl.create_some_context(interactive=False, answers=[0,0])

def create_queue(ctx) : return _cl.CommandQueue(ctx)

def create_context_and_queue(answers=[0,0], verbose=True):
      # Verbose the choice of platform and device
      if verbose:
            platform = _cl.get_platforms()[answers[0]]
            print('=' * 60)
            print('Platform - ID:  ' + str(answers[0]))
            print('Platform - Name:  ' + platform.name)
            print('Platform - Vendor:  ' + platform.vendor)
            print('Platform - Version:  ' + platform.version)
            print('Platform - Profile:  ' + platform.profile)

            device = platform.get_devices()[answers[1]]
            print('    ' + '-' * 56)
            print('    Device - ID:  ' \
                  + str(answers[1]))
            print('    Device - Name:  ' \
                  + device.name)
            print('    Device - Type:  ' \
                  + _cl.device_type.to_string(device.type))
            print('    Device - Max Clock Speed:  {0} Mhz'\
                  .format(device.max_clock_frequency))
            print('    Device - Compute Units:  {0}'\
                  .format(device.max_compute_units))
            print('    Device - Local Memory:  {0:.0f} KB'\
                  .format(device.local_mem_size/1024.0))
            print('    Device - Constant Memory:  {0:.0f} KB'\
                  .format(device.max_constant_buffer_size/1024.0))
            print('    Device - Global Memory: {0:.0f} GB'\
                  .format(device.global_mem_size/1073741824.0))
            print('    Device - Max Buffer/Image Size: {0:.0f} MB'\
                  .format(device.max_mem_alloc_size/1048576.0))
            print('    Device - Max Work Group Size: {0:.0f}'\
                  .format(device.max_work_group_size))
            print('=' * 60)

      # Create the runtime
      runtime = {}

      # Create the context
      runtime['ctx'] = create_context(answers=answers)

      # Createt he queue
      runtime['queue'] = create_queue(runtime['ctx'])

      # Memflags
      runtime['mf'] = _cl.mem_flags

      # Build options
      build_options = '-I {:}'.format(_files('openlc').joinpath('c_src'))

      # Now compile for the chosen device
      if verbose: print('Building kernels...', flush=True)

      # Build LC kernel
      if verbose: print('\tBuilding lc kernels... ', end='', flush=True)
      lc_prg = prg = _cl.Program(runtime['ctx'], _files('openlc').joinpath('c_src/lc.c').read_text() + '\n'\
                  + _files('openlc').joinpath('c_src/kepler.c').read_text() + '\n'\
                  + _files('openlc').joinpath('c_src/flux_drop.c').read_text() + '\n'\
            + _files('openlc').joinpath('c_src/sampler.c').read_text()).build(options=build_options)
      runtime['kernel_lc'] = lc_prg.lc
      runtime['kernel_reduce'] = lc_prg.reduce

      if verbose: print('OK', flush=True)

      # Build RV kernel
      if verbose: print('\tBuilding rv kernels... ', end='', flush=True)
      lc_prg = prg = _cl.Program(runtime['ctx'], _files('openlc').joinpath('c_src/rv.c').read_text() + '\n'\
                  + _files('openlc').joinpath('c_src/kepler.c').read_text()).build(options=build_options)
      runtime['kernel_rv1'] = lc_prg.rv1
      runtime['kernel_rv2'] = lc_prg.rv2

      if verbose: print('OK', flush=True)
      if verbose : print('=' * 60)

      return runtime


def print_device_info() :
    print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
    platform_id = 0
    for platform in _cl.get_platforms():
        print('=' * 60)
        print('Platform - ID:  ' + str(platform_id))
        print('Platform - Name:  ' + platform.name)
        print('Platform - Vendor:  ' + platform.vendor)
        print('Platform - Version:  ' + platform.version)
        print('Platform - Profile:  ' + platform.profile)
        device_count = 0
        for device in platform.get_devices():
            print('    ' + '-' * 56)
            print('    Device - ID:  ' \
                  + str(device_count))
            print('    Device - Name:  ' \
                  + device.name)
            print('    Device - Type:  ' \
                  + _cl.device_type.to_string(device.type))
            print('    Device - Max Clock Speed:  {0} Mhz'\
                  .format(device.max_clock_frequency))
            print('    Device - Compute Units:  {0}'\
                  .format(device.max_compute_units))
            print('    Device - Local Memory:  {0:.0f} KB'\
                  .format(device.local_mem_size/1024.0))
            print('    Device - Constant Memory:  {0:.0f} KB'\
                  .format(device.max_constant_buffer_size/1024.0))
            print('    Device - Global Memory: {0:.0f} GB'\
                  .format(device.global_mem_size/1073741824.0))
            print('    Device - Max Buffer/Image Size: {0:.0f} MB'\
                  .format(device.max_mem_alloc_size/1048576.0))
            print('    Device - Max Work Group Size: {0:.0f}'\
                  .format(device.max_work_group_size))
            device_count +=1
        platform_id +=1
    print('\n')