#include <kepler.h>

double __rv1(const double time,
            const double t_zero, const double period,
            const double K1,
            const double e, const double w, const double incl,
            const double V0,
            const int accurate_tp)
{
    // Get the True anomaly
    double nu = getTrueAnomaly(time, e, w, 
                                period,  t_zero, incl, accurate_tp);

    // Get the RV
    return K1*(cos(nu + w) + e*cos(w)) + V0;
}

double __rv2(const double time, double * RV1, double * RV2,
            const double t_zero, const double period,
            const double K1, const double K2,
            const double e, const double w, const double incl,
            const double V0,
            const int accurate_tp)
{
    // Get the True anomaly
    double nu = getTrueAnomaly(time, e, w, 
                                period,  t_zero, incl, accurate_tp);

    // Get the RV
    *RV1 = K1*(cos(nu + w) + e*cos(w)) + V0;
    *RV2 = K2*(cos(nu + w + M_PI) + e*cos(w)) + V0;
}

__kernel void rv1(
    __global const double *a_g, __global double *res_g, 
    const double t_zero, const double period,
    const double K1, 
    const double e, const double w, const double incl,
    const double V0,
    const int accurate_tp)
{
  int gid = get_global_id(0);
  res_g[gid] = __rv1(a_g[gid],
                    t_zero, period,
                    K1,
                    e,w,incl,
                    V0,
                    accurate_tp);
}

__kernel void rv2(
    __global const double *a_g, __global double *res_g1,  __global double *res_g2, 
    const double t_zero, const double period,
    const double K1, const double K2, 
    const double e, const double w, const double incl,
    const double V0,
    const int accurate_tp)
{
  int gid = get_global_id(0);
  double rv1_value, rv2_value;
  __rv2(a_g[gid], &rv1_value, &rv2_value,
                t_zero, period,
                K1, K2,
                e,w,incl,
                V0,
                accurate_tp);
  res_g1[gid] = rv1_value;
  res_g2[gid] = rv2_value;
}