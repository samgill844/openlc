double clip(double a, double b, double c);
double q1(double z, double p, double c, double a, double g, double I_0);
double q2(double z, double p, double c, double a, double g, double I_0, double eps);
double flux_drop_analytical_power_2(double d_radius, double k, double c, double a, double eps);
double flux_drop_analytical_uniform(double d_radius, double k, double SBR);
double get_intensity_from_limb_darkening_law (int ld_law, double * ldc, double mu_i, int offset);
double flux_drop_annulus(double d_radius, double k, double SBR, int ld_law, double * ldc, int n_annulus, int primary, int offset);
double area(double z, double r1, double r2);