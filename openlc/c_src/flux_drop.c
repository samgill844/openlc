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
	double J2 = a*c*pow(sa,(g-1))*pow(p,4)*(0.125*aq + (1./12.)*q*(q*q-2.5)*sqrt(clip(1-q*q,0.0,1.0)) );
	double d0 = 1 - c + c*pow(sb,g);
	double d1 = -a*c*rb*pow(sb,(g-1));
	double K1 = (d0-rb*d1)*acos(d) + ((rb*d+(2./3.)*(1-d*d))*d1 - d*d0)*sqrt(clip(1-d*d,0.0,1.0));
	double K2 = (1/3)*c*a*pow(sb,(g+0.5))*(1-d);
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
