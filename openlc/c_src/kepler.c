
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
    double eps = 1e-5;
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