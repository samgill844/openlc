double _loglike(const double y, const double yerr, const double model, const double jitter, const int offset);

double _delta(const double th, const double sin2i, const double omrad, const double ecc);
double __delta (const double th, const double * z0);
double vsign(double a, double b);
double merge(double a , double b, bool mask);
double brent_minimum_delta(double ax, double bx, double cx, const double * z0); 
double t_ecl_to_peri(const double t_zero, const double P, const double incl, const double ecc, const double omrad);
double getEccentricAnomaly(double M, const double ecc);
double getTrueAnomaly(const double time, const double e, const double w, const double period,  const double t_zero, const double incl, const int accurate_tp);
double get_z(const double nu, const double e, const double incl, const double w, const double radius_1);
double getProjectedPosition(const double nu, const double w, const double incl);

double clip(double a, double b, double c);
double q1(double z, double p, double c, double a, double g, double I_0);
double q2(double z, double p, double c, double a, double g, double I_0, double eps);
double flux_drop_analytical_power_2(double d_radius, double k, double c, double a, double eps);
double flux_drop_analytical_uniform(double d_radius, double k, double SBR);
double get_intensity_from_limb_darkening_law (int ld_law, double * ldc, double mu_i, int offset);
double area(double z, double r1, double r2);
double flux_drop_annulus(double d_radius, double k, double SBR, int ld_law, double * ldc, int n_annulus, int primary, int offset);
double __lc (const double time,
    const double t_zero, const double period,
    const double radius_1, const double k,const double incl,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp);
double _lc (const double time,
    const double t_zero, const double period,
    const double radius_1, const double k,const double incl,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp);
double __rv1(const double time,
            const double t_zero, const double period,
            const double K1,
            const double e, const double w, const double incl,
            const double V0,
            const int accurate_tp);
void __rv2(const double time, double * RV1, double * RV2,
            const double t_zero, const double period,
            const double K1, const double K2,
            const double e, const double w, const double incl,
            const double V0,
            const int accurate_tp);
double transit_width(double radius_1, double k, double b, double period);



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
















double _delta(const double th, const double sin2i, const double omrad, const double ecc) { return (1-pow(ecc,2))*(sqrt(1-sin2i*pow(sin(th+omrad),2))/(1+ecc*cos(th)));}
double __delta (const double th, const double * z0) {return _delta(th, z0[0], z0[1], z0[2]);}

double vsign(double a, double b)
{
    if (b >= 0) return fabs(a);
    return -fabs(a);
}
 
double merge(double a , double b, bool mask)
{
    if (mask) return a;
    return b;
}

double brent_minimum_delta(double ax, double bx, double cx, const double * z0)
{
    //#######################################################
    //# Find the minimum of a function (func) between ax and
    //# cx and that func(bx) is less than both func(ax) and 
    //# func(cx). 
    //# z0 is the additional arguments 
    //#######################################################
    //# pars
    double tol = 1e-5;
    int itmax = 100;
    //double eps = 1e-5;
    double cgold = 0.3819660;
    double zeps = 1.0E-10;

    double a = min(ax, cx);
    double b = max(ax, cx);
    double d = 0.;
    double v = bx;
    double w = v;
    double x = v;
    double e = 0.0;
    double fx = __delta(x,z0);
    double fv = fx;
    double fw = fx;
    int iter;

    double xm, tol1, tol2, r, q,p, etemp, fu, u;
    for (iter=0; iter <itmax; iter++)
    {
        xm = 0.5*(a+b);
        tol1 = tol*fabs(x)+zeps;
        tol2 = 2.0*tol1;
        if(fabs(x-xm) <= (tol2-0.5*(b-a))) return x;
        
        if(fabs(e) > tol1)
        {
            r = (x-w)*(fx-fv);
            q = (x-v)*(fx-fw);
            p = (x-v)*q - (x-w)*r;
            q = 2.0*(q-r);
            if (q > 0.0)  p = - p;
            q = fabs(q);
            etemp = e;
            e = d;

            if (  (fabs(p) >= fabs(.5*q*etemp)) || (p <= q*(a-x)) || (p >= q*(b-x)))
            {
                e = merge(a-x, b-x, p >= q*(b-x));
                d = cgold*e;
            }
            else
            {
                d = p/q;
                u=x+d;
                if ( ((u-a) < tol2) || ((b-u) < tol2))  d = vsign(tol1, xm-x);
            }
        }
        else
        {
            e = merge(a-x, b-x, x >= xm);
            d = cgold*e;
        }

        u = merge(x+d, x+vsign(tol1,d), fabs(d) >= tol1);
        fu = __delta(u,z0);


        if (fu <= fx)
        {
            if ( u >= x)  a = x;
            else  b = x ;
            v = w ;
            w = x ;
            x = u ;
            fv = fw;
            fw = fx ;
            fx = fu ;
        }
        else
        {
            if (u < x)  a = u ;
            else  b = u ;
            if ((fu <= fw) || (w==x))
            {
                v=w;
                fv=fw;
                w=u;
                fw=fu;
            }
            else if ((fu <= fv) || (v==x) || (v==w))
            {
                v = u;
                fv = fu; 
            }
        }
    }
    return 1. ;
}



double t_ecl_to_peri(const double t_zero, const double P, const double incl, const double ecc, const double omrad)
{
    double sini = sin(incl);
    double sin2i = pow(sini,2);
    double theta = 0.5*M_PI-omrad;
    double ta = theta-0.125*M_PI;
    double tb = theta;
    double tc = theta+0.125*M_PI;
    double E;

    double z0[3] = {sin2i, omrad, ecc};
    theta = brent_minimum_delta(ta, tb, tc, z0);
    if (theta == M_PI) E = M_PI;
    else E = 2*atan(sqrt((1.-ecc)/(1.+ecc))*tan(theta/2.));
    return t_zero - (E - ecc*sin(E))*P/(2*M_PI);

}

double getEccentricAnomaly(double M, const double ecc)
{
    M = fmod(M , 2*M_PI);
    int flip = 0;
    if (ecc == 0) return M;
    if (M > M_PI)
    {
        M = 2*M_PI - M;
        flip = 1;
    }

    double alpha = (3.*M_PI + 1.6*(M_PI-fabs(M))/(1.+ecc) )/(M_PI - 6./M_PI);
    double d = 3.*(1 - ecc) + alpha*ecc;
    double r = 3.*alpha*d * (d-1+ecc)*M + pow(M,3.);
    double q = 2.*alpha*d*(1-ecc) - pow(M,2.);
    double w = pow((fabs(r) + sqrt(pow(q,3.) + pow(r,2.))),(2./3.));
    double E = (2.*r*w/(pow(w,2.) + w*q + pow(q,2.)) + M) / d;
    double f_0 = E - ecc*sin(E) - M;
    double f_1 = 1. - ecc*cos(E);
    double f_2 = ecc*sin(E);
    double f_3 = 1.-f_1;
    double d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1);
    double d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (pow(d_3,2))*f_3/6);
    E = E -f_0/(f_1 + 0.5*d_4*f_2 + pow(d_4,2.)*f_3/6 - pow(d_4,3.)*f_2/24);
    if (flip==1) E =  2*M_PI - E;
    return E;
}

double getTrueAnomaly(const double time, const double e, const double w, const double period,  const double t_zero, const double incl, const int accurate_tp)
{
    double tp;
    if (accurate_tp==1) tp = t_ecl_to_peri(t_zero,period,incl,e,w);
    else tp = 0.;

    if (e<1e-5) 
    {
        return ((time - tp)/period - floor(((time - tp)/period)))*2.*M_PI;
    }
    else
    {
        // Calcualte the mean anomaly
        double M = 2*M_PI*fmod((time -  tp  )/period , 1.);

        // Calculate the eccentric anomaly
        double E = getEccentricAnomaly(M, e);
        
        // Now return the true anomaly
        return 2.*atan(sqrt((1.+e)/(1.-e))*tan(E/2.));
    }
}

double get_z(const double nu, const double e, const double incl, const double w, const double radius_1) {return (1-pow(e,2)) * sqrt( 1.0 - pow(sin(incl),2)  *  pow(sin(nu + w),2)) / (1 + e*cos(nu)) /radius_1;}

double getProjectedPosition(const double nu, const double w, const double incl) { return sin(nu+w)*sin(incl);}

















double clip(double a, double b, double c)
{
	if (a < b)
		return b;
	else if (a > c)
		return c;
	else
		return a;
}



/* Analytical Power 2 law (Maxted & Gill 2019)*/

double q1(double z, double p, double c, double a, double g, double I_0)
{
	double zt = clip(fabs(z), 0,1-p);
	double s = 1-zt*zt;
	double c0 = (1-c+c*pow(s,g));
	double c2 = 0.5*a*c*pow(s,(g-2))*((a-1)*zt*zt-1);
	return -I_0*M_PI*p*p*(c0 + 0.25*p*p*c2 - 0.125*a*c*p*p*pow(s,(g-1)));
}

double q2(double z, double p, double c, double a, double g, double I_0, double eps)
{
	double zt = clip(fabs(z), 1-p,1+p);
	double d = clip((zt*zt - p*p + 1)/(2*zt),0,1);
	double ra = 0.5*(zt-p+d);
	double rb = 0.5*(1+d);
	double sa = clip(1-ra*ra,eps,1);
	double sb = clip(1-rb*rb,eps,1);
	double q = clip((zt-d)/p,-1,1);
	double w2 = p*p-(d-zt)*(d-zt);
	double w = sqrt(clip(w2,eps,1));
	double c0 = 1 - c + c*pow(sa,g);
	double c1 = -a*c*ra*pow(sa,(g-1));
	double c2 = 0.5*a*c*pow(sa,(g-2))*((a-1)*ra*ra-1);
	double a0 = c0 + c1*(zt-ra) + c2*(zt-ra)*(zt-ra);
	double a1 = c1+2*c2*(zt-ra);
	double aq = acos(q);
	double J1 =  (a0*(d-zt)-(2./3.)*a1*w2 + 0.25*c2*(d-zt)*(2.0*(d-zt)*(d-zt)-p*p))*w + (a0*p*p + 0.25*c2*pow(p,4))*aq ;
    if (J1<-1 || J1>1) J1=0;
    //J1 = clip(J1,0,0.3); // This is unstable
	double J2 = a*c*pow(sa,(g-1))*pow(p,4)*(0.125*aq + (1./12.)*q*(q*q-2.5)*sqrt(clip(1-q*q,0.0,1.0)) );
	double d0 = 1 - c + c*pow(sb,g);
	double d1 = -a*c*rb*pow(sb,(g-1));
	double K1 = (d0-rb*d1)*acos(d) + ((rb*d+(2./3.)*(1-d*d))*d1 - d*d0)*sqrt(clip(1-d*d,0.0,1.0));
	double K2 = (1/3)*c*a*pow(sb,(g+0.5))*(1-d);
    //printf("\nQ2 func %f %f %f %f %f", I_0, J1 , J2 , K1 , K2);
	return -I_0*(J1 - J2 + K1 - K2);
}

double flux_drop_analytical_power_2(double d_radius, double k, double c, double a, double eps)
{
	/*
	Calculate the analytical flux drop por the power-2 law.
	
	Parameters
	d_radius : double
		Projected seperation of centers in units of stellar radii.
	k : double
		Ratio of the radii.
	c : double
		The first power-2 coefficient.
	a : double
		The second power-2 coefficient.
	f : double
		The flux from which to drop light from.
	eps : double
		Factor (1e-9)
	*/
	double I_0 = (a+2)/(M_PI*(a-c*a+2));
	double g = 0.5*a;

	if (d_radius < 1-k) return q1(d_radius, k, c, a, g, I_0);
	else if (fabs(d_radius-1) < k) return q2(d_radius, k, c, a, g, I_0, 1e-9);
	else return 0;
}


/* Analytical uniform flux */
double flux_drop_analytical_uniform(double d_radius, double k, double SBR)
{
		if(d_radius >= 1. + k)
			return 0.0;		//no overlap
		if(d_radius >= 1. && d_radius <= k - 1.) 
			return 0.0;     //total eclipse of the star
		else if(d_radius <= 1. - k) 
		{
			if (SBR !=-99) return 1 - SBR*k*k;	//planet is fully in transit
			else  return - k*k;	//planet is fully in transit
		}
		else						//planet is crossing the limb
		{
			double kap1=acos(fmin((1. - k*k + d_radius*d_radius)/2./d_radius, 1.));
			double kap0=acos(fmin((k*k + d_radius*d_radius - 1.)/2./k/d_radius, 1.));
			if (SBR != -99) return - SBR*  (k*k*kap0 + kap1 - 0.5*sqrt(fmax(4.*d_radius*d_radius - powf(1. + d_radius*d_radius - k*k, 2.), 0.)))/M_PI;
			else
				return - (k*k*kap0 + kap1 - 0.5*sqrt(fmax(4.*d_radius*d_radius - powf(1. + d_radius*d_radius - k*k, 2.), 0.)))/M_PI;

		}
}




/* Annulus integration */
double get_intensity_from_limb_darkening_law (int ld_law, double * ldc, double mu_i, int offset)
{
	/*
	Calculte limb-darkening for a variety of laws e.t.c.
	[0] linear (Schwarzschild (1906, Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen. Mathematisch-Physikalische Klasse, p. 43)
	[1] Quadratic Kopal (1950, Harvard Col. Obs. Circ., 454, 1)
	[2] Square-root (Díaz-Cordovés & Giménez, 1992, A&A, 259, 227) 
	[3] Logarithmic (Klinglesmith & Sobieski, 1970, AJ, 75, 175)
	[4] Exponential LD law (Claret & Hauschildt, 2003, A&A, 412, 241)
	[5] Sing three-parameter law (Sing et al., 2009, A&A, 505, 891)
	[6] Claret four-parameter law (Claret, 2000, A&A, 363, 1081)
	[7] Power-2 law (Maxted 2018 in prep)
	*/
	if (ld_law == 0) 
		return 1 - ldc[offset]*(1 - mu_i);
	if (ld_law == 1) 
		return 1 - ldc[offset]*(1 - mu_i) - ldc[offset+1] * powf((1 - mu_i),2)  ;
	if (ld_law == 2) 
		return 1 -  ldc[offset]*(1 - mu_i) - ldc[offset+1]*(1 - powf(mu_i,2) ) ;
	if (ld_law == 3) 
		return 1 -  ldc[offset]*(1 - mu_i) - ldc[offset+1]*mu_i*logf(mu_i); 
	if (ld_law == 4) 
		return 1 -  ldc[offset]*(1 - mu_i) - ldc[offset+1]/(1-expf(mu_i));  
	if (ld_law == 5) 
		return 1 -  ldc[offset]*(1 - mu_i) - ldc[offset+1]*(1 - powf(mu_i,1.5)) - ldc[offset+2]*(1 - powf(mu_i,2));
	if (ld_law == 6) 
		return 1 - ldc[offset]*(1 - powf(mu_i,0.5)) -  ldc[offset+1]*(1 - mu_i) - ldc[offset+2]*(1 - powf(mu_i,1.5))  - ldc[offset+3]*(1 - powf(mu_i,2));
	if (ld_law == 7) 
		return 1 - ldc[offset]*(1 - powf(mu_i,ldc[offset+1]));	
	else
		return 1 - ldc[offset]*(1 - mu_i);
}

double area(double z, double r1, double r2)
{
	//
	// Returns area of overlapping circles with radii x and R; separated by a distance d
	//

	double arg1 = clip((z*z + r1*r1 - r2*r2)/(2.*z*r1),-1,1);
	double arg2 = clip((z*z + r2*r2 - r1*r1)/(2.*z*r2),-1,1);
	double arg3 = clip(max((-z + r1 + r2)*(z + r1 - r2)*(z - r1 + r2)*(z + r1 + r2), 0.),-1,1);

	if   (r1 <= r2 - z) return M_PI*r1*r1;							                              // planet completely overlaps stellar circle
	else if (r1 >= r2 + z) return M_PI*r2*r2;						                                  // stellar circle completely overlaps planet
	else                return r1*r1*acosf(arg1) + r2*r2*acosf(arg2) - 0.5*sqrtf(arg3);          // partial overlap
}

double flux_drop_annulus(double d_radius, double k, double SBR, int ld_law, double * ldc, int n_annulus, int primary, int offset)
{

	double dr = 1.0/n_annulus;

	int ss;
	double r_ss, mu_ss, ra, rb, I_ss, F_ss, fp,A_ra_rc , A_rb_rc, A_annuli_covered, A_annuli, Flux_total, Flux_occulted;
	Flux_total = 0.0;
	Flux_occulted = 0.0;
	double f = 0.;

	for (ss=0; ss < n_annulus;ss++)
	{
		// Calculate r_ss
		r_ss = (ss + 0.5)/n_annulus;

		ra = r_ss + 0.5/n_annulus;
		rb = r_ss - 0.5/n_annulus;

		// Calculate mu_ss
		mu_ss = sqrt(1 - r_ss*r_ss);

		// Calculate upper (ra) and lower extent (rb)
		if (primary==0)
		{
			// Get intensity from ss
			I_ss = get_intensity_from_limb_darkening_law(ld_law, ldc, mu_ss, offset);

			// Get flux at mu_ss
			F_ss = I_ss*2*M_PI*r_ss*dr;

			if ((ra + k) < d_radius) fp = 0;
			else if (rb >= (d_radius-k) & ra <= (d_radius + k))
			{
				// Calculate intersection between the circle of star 2
				// and the outer radius of the annuli, (ra)
				A_ra_rc = area(d_radius, k, ra);

				// Calculate intersection between the circle of star 2
				// and the inner radius of the annuli, (rb)
				A_rb_rc = area(d_radius, k, rb);

				// So now the area of the of the anuuli covered by star 2 
				// is the difference between these two areas...
				A_annuli_covered = A_ra_rc - A_rb_rc;

				// Great, now we need the area of the annuli itself...
				A_annuli = M_PI*(ra*ra - rb*rb);

				// So now we can calculate fp, 
				fp = A_annuli_covered/A_annuli;	
			}
			else
				fp = 0.0;

		}
		else
		{
			// Get intensity at mu_ss
			I_ss = get_intensity_from_limb_darkening_law(ld_law, ldc, mu_ss,offset);

			// Get Flux at mu_ss
			F_ss = I_ss*2*M_PI*r_ss*dr;

			if   ((d_radius + k) <= 1.0)  fp = 1;   // all the flux from star 2 is occulted as the 
												    // annulus sits behind star 1
			else if ((d_radius - k) >= 1.0)  fp = 0;  // All the flux from star 2 is visible
			else if ((d_radius + ra) <= 1.0)  fp = 1; // check that the annulus is not entirely behind star 1
			else if ((d_radius - ra) >= 1.0)  fp = 0; // check that the annulus is not entirely outside star 1
			else
			{
				// Calculate intersection between the circle of star 2
				// and the outer radius of the annuli, (ra)
				A_ra_rc = area(d_radius, 1.0, ra*k);

				// Calculate intersection between the circle of star 2
				// and the inner radius of the annuli, (rb)
				A_rb_rc = area(d_radius, 1.0, rb*k);


				// So now the area of the of the anuuli covered by star 2 
				// is the difference between these two areas...
				A_annuli_covered = A_ra_rc - A_rb_rc;

				// Great, now we need the area of the annuli itself...
				A_annuli = M_PI*((ra*k)*(ra*k) - (rb*k)*(rb*k));

				// So now we can calculate fp, 
				fp = A_annuli_covered/A_annuli;
			}

		}

		// Now we can calculate the occulted flux...
		Flux_total =  Flux_total + F_ss;
		Flux_occulted =  Flux_occulted + F_ss*fp;
	}


	if (primary==0) return f - Flux_occulted/Flux_total;
	else
		return f - k*k*SBR*Flux_occulted/Flux_total;

}













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

  //double ldc[2]={c,alpha};

  // Check the distance between them to see if they are transiting
  if (z < (1.0+k))
  {
    // Let's find out if its a primary or secondary
    double f = getProjectedPosition(nu, w , incl);
    if (f>0)
    {
      //if (ld_law==0) F_transit = flux_drop_analytical_uniform(z, k, -99);
      //if (ld_law==1) F_transit = flux_drop_annulus(z, k, 1, 7, ldc, 4000, 0, 0);
      if (ld_law==2) F_transit = flux_drop_analytical_power_2(z, k, c, alpha, 0.001);
      //if (ld_law==-1) F_transit = -k;
      if ((ld_law==-2) &&  (fabs((time - t_zero) / period) < 0.5)) F_transit = flux_drop_analytical_power_2(z, k, c, alpha, 0.001);
    }
    //printf("\nNu %f   z  %f", f, F_transit);
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


__kernel void reduce ( __global const double *input, 
                         __global double *partialSums,
                         __local double *localSums)
 {
  uint local_id = get_local_id(0);
  uint group_size = get_local_size(0);

  // Copy from global to local memory
  localSums[local_id] = input[get_global_id(0)];

  // Loop for computing localSums : divide WorkGroup into 2 parts
  for (uint stride = group_size/2; stride>0; stride /=2)
     {
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_id < stride)
        localSums[local_id] += localSums[local_id + stride];
     }

  // Write result into partialSums[nWorkGroups]
  if (local_id == 0)
    partialSums[get_group_id(0)] = localSums[0];
 } 


 __kernel void reduce_loglike ( __global const double *input, 
                         __global double *partialSums,
                         __local double *localSums)
 {
  uint local_id = get_local_id(0);
  uint group_size = get_local_size(0);

  // Copy from global to local memory
  localSums[local_id] = input[get_global_id(0)]*2;

  // Loop for computing localSums : divide WorkGroup into 2 parts
  for (uint stride = group_size/2; stride>0; stride /=2)
     {
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_id < stride)
        localSums[local_id] += localSums[local_id + stride];
     }

  // Write result into partialSums[nWorkGroups]
  if (local_id == 0)
    partialSums[get_group_id(0)] = localSums[0];
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


__kernel void lc_loglike(
    __global const double *time_g, __global double *flux_g, __global double *flux_err_g,  __global double *partialSums, __local double *localSums, const int size,
    const double t_zero, const double period,
    const double radius_1, const double k,const double incl,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp,
    const double jitter, const int offset)
{
  uint gid = get_global_id(0);
  uint local_id = get_local_id(0);
  uint group_size = get_local_size(0);

  // Get the first model
  if (gid < size)
  {
    double model = _lc(time_g[gid],
                    t_zero, period,
                    radius_1, k, incl,
                    e,w,
                    c,alpha,
                    cadence, noversample,
                    light_3,
                    ld_law,
                    accurate_tp);

    // Copy from global to local memory
    localSums[local_id] = _loglike(flux_g[gid], flux_err_g[gid], model, jitter, offset);
  }
  else
  {
    localSums[local_id] = 0.;
  }
  // printf("\n%f    %f     %f    %u", time_g[gid], flux_g[gid], flux_err_g[gid], gid);
  // Loop for computing localSums : divide WorkGroup into 2 parts
  for (uint stride = group_size/2; stride>0; stride /=2)
     {
      // Waiting for each 2x2 addition into given workgroup
      barrier(CLK_LOCAL_MEM_FENCE);

      // Add elements 2 by 2 between local_id and local_id + stride
      if (local_id < stride)
        localSums[local_id] += localSums[local_id + stride];
     }

  // Write result into partialSums[nWorkGroups]
  if (local_id == 0)
    partialSums[get_group_id(0)] = localSums[0];
 } 








__kernel void template_match(
    __global const double *time_trial_g, __global double *DeltaL_trial_g, 
    __global const double *time_g, __global double *flux_g, __global double *flux_err_g,  __global double *normalisation_model, 
    const int size_trial, const int size,
    const double width,
    const double period,
    const double radius_1, const double k,const double incl,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp,
    const double jitter, const int offset)
{
    int gid = get_global_id(0);

    // Get the first model
    double model;
    for (int x=0; x < size; x++)
    {
        //if (gid==20 && x<200){
        if (fabs(time_trial_g[gid] - time_g[x]) < (width/2))
        {
        model = _lc(time_g[x],
                        time_trial_g[gid], period,
                        radius_1, k, incl,
                        e,w,
                        c,alpha,
                        cadence, noversample,
                        light_3,
                        ld_law,
                        accurate_tp);

        // Copy from global to local memory
        DeltaL_trial_g[gid] += (_loglike(flux_g[x]/normalisation_model[x], flux_err_g[x]/normalisation_model[x], model, jitter, offset) - _loglike(flux_g[x]/normalisation_model[x], flux_err_g[x]/normalisation_model[x], 1., jitter, offset));
        }
        
    }
}


double transit_width(double radius_1, double k, double b, double period)
{
    return period*asin(radius_1*sqrt( ((1+k)*(1+k)-b*b) / (1-b*b*radius_1*radius_1) ))/M_PI;
}

__kernel void template_match_batch(
    __global const double *time_trial_g, __global double *DeltaL_trial_g, 
    __global const double *time_g, __global double *flux_g, __global double *flux_err_g,  __global double *normalisation_model, 
    const int size_trial, const int size,
    const double period,
    __global const double * radius_1, __global const double * k, __global const double * b,
    const int size_radius_1, const int size_k,const int size_b,
    const double e, const double w,
    const double c, const double alpha,
    const double cadence, const int noversample,
    const double light_3,
    const int ld_law,
    const int accurate_tp,
    const double jitter, const int offset)
{
    int gid = get_global_id(0);

    // Get the first model
    double model, incl,width;
    int count=0;
    for (int y=0; y < size_radius_1; y++)
    {
        for (int z=0; z < size_k; z++)
        {
            for (int q=0; q < size_b; q++)
            {
                for (int x=0; x < size; x++)
                {
                    // Get the width
                    incl = acos(radius_1[y]*b[q]);
                    width = transit_width(radius_1[y], k[z], b[q], period);
                    //printf("\n%f", width);
                    if (fabs(time_trial_g[gid] - time_g[x]) < (width/2))
                    {
                        model = _lc(time_g[x],
                                        time_trial_g[gid], period,
                                        radius_1[y], k[z], incl,
                                        e,w,
                                        c,alpha,
                                        cadence, noversample,
                                        light_3,
                                        ld_law,
                                        accurate_tp);

                        // Copy from global to local memory (gid*size_radius_1*size_k*size_b)  + count
                        DeltaL_trial_g[gid*size_radius_1*size_k*size_b + count] += (_loglike(flux_g[x]/normalisation_model[x], flux_err_g[x]/normalisation_model[x], model, jitter, offset) - _loglike(flux_g[x]/normalisation_model[x], flux_err_g[x]/normalisation_model[x], 1., jitter, offset));
                    }
                }
                count++;
            }
        }
    }
}



__kernel void bin_data(
    __global const double *time, __global const double *flux, const int time_size,
    __global const double *edges, __global double *binned_flux, __global double *binned_flux_err, const int binned_size,
    __global int * count)
{
    int gid = get_global_id(0);
    if (gid > binned_size) return ;

    // Now loop the time axis
    // binned_flux[gid] is going to be edges[gid]  < binned_flux[gid] < edges[gid+1] 
    double tmp[100];
    for (int i=0; i < time_size; i++)
    {
        if (time[i] > edges[gid] && time[i] < edges[gid+1] && count[gid] < 100) 
        {
            tmp[count[gid]+1] = flux[i];
            count[gid]++;
        }
        if (time[i] > edges[gid+1]) break ;
    }
    
    
    // OK, now check
    if (count[gid]>0)
    {
        // Lets do the mean
        for (int i=1; i< (count[gid]+1); i++)
        {
            binned_flux[gid] += tmp[i];
        }
        binned_flux[gid] = binned_flux[gid] / count[gid];

        // Now lets do the standard deviation
        for (int i=1; i< (count[gid]+1); i++)
        {
            binned_flux_err[gid] += pow(tmp[i] - binned_flux[gid], 2);
        }
        binned_flux_err[gid] = sqrt(binned_flux_err[gid]/count[gid]) / sqrt((double) count[gid]);
    }
    
}














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

void __rv2(const double time, double * RV1, double * RV2,
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


__kernel void check_proximity_of_timestamps(__global const double *x_trial, __global const double *x_ref, const int x_ref_size, const double width,
                                            __global int * mask)
{
    uint gid = get_global_id(0);
    mask[gid] = 0;
    for (int i=0; i<x_ref_size; i++)
    {
        if (fabs(x_trial[gid] - x_ref[i]) < width) 
        {
            mask[gid] = 1;
            break;
        }
    }
}