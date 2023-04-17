float test_function(float in);
double getTrueAnomaly(const double time, const double e, const double w, const double period,  const double t_zero, const double incl, const int accurate_tp);
double t_ecl_to_peri(const double t_zero, const double P, const double incl, const double ecc, const double omrad);
double getEccentricAnomaly(double M, const double ecc);
double get_z(const double nu, const double e, const double incl, const double w, const double radius_1);

double _delta(const double th, const double sin2i, const double omrad, const double ecc);
double __delta (const double th, const double * z0);
double vsign(double a, double b);
double merge(double a , double b, bool mask);
double brent_minimum_delta(double ax, double bx, double cx, const double * z0);
double getProjectedPosition(const double nu, const double w, const double incl);