
#include <kepler.h>
#include <flux_drop.h>
#include <sampler.h>


double __lc (const double time,
    const double t_zero, const double period,
    const double radius_1, const double k,const double incl,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp)
{
  // Get the True anomal
  double nu = getTrueAnomaly(time, e, w, 
                              period,  t_zero, incl, accurate_tp);

  // Get the projected seperation
  double z = get_z(nu, e, incl, w, radius_1);

  // Allocate the flux
  double F_transit = 0;

  double ldc[2]={c,alpha};

  // Check the distance between them to see if they are transiting
  if (z < (1.0+k))
  {
    // Let's find out if its a primary or secondary
    double f = getProjectedPosition(nu, w , incl);
    if (f>0)
    {
      if (ld_law==0) F_transit = flux_drop_analytical_uniform(z, k, -99);
      if (ld_law==1) F_transit = flux_drop_annulus(z, k, 1, 7, ldc, 4000, 0, 0);
      if (ld_law==2) F_transit = flux_drop_analytical_power_2(z, k, c, alpha, 0.001);
      if (ld_law==-1) F_transit = -k;
      if ((ld_law==-2) &&  (fabs((time - t_zero) / period) < 0.5)) F_transit = flux_drop_analytical_power_2(z, k, c, alpha, 0.001);
    }
  }

  // Put the model together
  double model = (1 + F_transit);

  // Now lets account for third light
  if (light_3>0) model = model/(1. + light_3) + (1.-1.0/(1. + light_3));
  
  return model;
}


double _lc (const double time,
    const double t_zero, const double period,
    const double radius_1, const double k,const double incl,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp)
{
  if (cadence>0)
  {
    double dr = (cadence/2) / ((noversample-1)/2);
    double model = 0. ;
    for (int i=0; i<noversample; i++)
    {
      model += __lc(time -dr*((noversample-1)/2) + i*dr,
                    t_zero, period,
                    radius_1, k, incl,
                    e,w,
                    c,alpha,
                    cadence, noversample,
                    light_3,
                    ld_law,
                    accurate_tp);
    }
    model /= noversample;
    return model;
  }
  else
  {
      return __lc(time,
                    t_zero, period,
                    radius_1, k, incl,
                    e,w,
                    c,alpha,
                    cadence, noversample,
                    light_3,
                    ld_law,
                    accurate_tp);
  }
}


__kernel void lc(
    __global const double *time_g, __global double *flux_g, 
    const double t_zero, const double period,
    const double radius_1, const double k,const double incl,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp)
{
  int gid = get_global_id(0);
  flux_g[gid] = _lc(time_g[gid],
                    t_zero, period,
                    radius_1, k, incl,
                    e,w,
                    c,alpha,
                    cadence, noversample,
                    light_3,
                    ld_law,
                    accurate_tp);
}

__kernel void reduce(__global int* x) {
   __local int a, b;
   a = 0;
   b = 0;
   /* Increment without atomic add */
   a++;
   /* Increment with atomic add */
   atomic_inc(&b);
   x[0] = a;
   x[1] = b;
}