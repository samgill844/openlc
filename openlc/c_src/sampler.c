double _loglike(const double y, const double yerr, const double model, const double jitter, const int offset);


double _loglike(const double y, const double yerr, const double model,
                const double jitter, const int offset)
{
    double wt = 1. / (yerr*yerr + jitter*jitter);
    double loglikeliehood = -0.5*((y - model)*(y - model)*wt);
    if (offset==1) loglikeliehood -= 0.5*log(wt);
    return loglikeliehood;
}


__kernel void loglike(__global double * result, __global const double *y_g, __global const double *yerr_g, __global const double *model_g,
                const double jitter, const int offset)
{
  int gid = get_global_id(0);
  result[gid] = _loglike(y_g[gid], yerr_g[gid], model_g[gid],
                jitter, offset);
}