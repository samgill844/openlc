__all__ = ['secondary_phase', 'transit_width', 'transit_width_full', 'stellar_density', 'planet_logg_2',
            'convert_width_and_period_to_radius_1', 'convert_depth_to_k', 'equilbrium_temperature', 'planet_albedos',
            'transmission_signal', 'atmospheric_scale_height', 'JWST_transmission_metric']


import numba as _numba, math as _math, numpy as _np
from astropy import constants as _c

mean_solar_day = 86400.002
R_SunN    = 6.957E8           # m, solar radius 
GM_SunN   = 1.3271244E20      # m3.s-2, solar mass parameter
V_SunN    = 4*_math.pi*R_SunN**3/3  # m3,  solar volume 
_arsun   = (GM_SunN*mean_solar_day**2/(4*_math.pi**2))**(1/3.)/R_SunN
_rhostar = 3*_math.pi*V_SunN/(GM_SunN*mean_solar_day**2)
R = _c.R.value
R_sun_earth = _c.R_sun.to(_c.R_earth).value
M_sun_earth = _c.M_sun.to(_c.M_earth).value
k_b = _c.k_B.value 
u_kg = _c.u.value

planet_albedos = {'Mercury' : 0.088,
                  'Venus' : 0.76,
                  'Earth' : 0.306,
                  'Moon' : 0.11,
                  'Mars' : 0.25,
                  'Jupiter' : 0.503,
                  'Saturn' : 0.342,
                  'Enceladus' : 0.8,
                  'Uranus' : 0.3,
                  'Neptune' : 0.29,
                  'Pluto' : 0.4,
                  'Eris' : 0.96}






@_numba.njit
def atmospheric_scale_height(Teff, g, M = 0.00207 ):
    # M = mean mass of one mol of atmospheric particles ((0.00207 for Saturn)
    # Returns atmospheric scale height in m
    return R*Teff/(M*g)

@_numba.njit
def secondary_phase(e, w):
    Phi = _math.pi + 2*_math.atan((e*_math.cos(w)) / _math.sqrt(1 - e**2))
    return (Phi - _math.sin(Phi)) / (2*_math.pi)

@_numba.njit
def transit_width(radius_1, k, b, period=1.):
    """
    Total ciurcular transit duration.
    See equation (3) from Seager and Malen-Ornelas, 2003ApJ...585.1038S.
    :param radius_1: R_star/a
    :param k: R_planet/R_star
    :param b: impact parameter = a.cos(i)/R_star
    :param P: orbital period (optional, default P=1)
    :returns: Total transit duration in the same units as P.
    """
    return  period*_math.asin(radius_1*_math.sqrt( ((1+k)**2-b**2) / (1-b**2*radius_1**2) ))/_math.pi

@_numba.njit
def transit_width_full(radius_1, k, b, period=1.):
    """
    Total ciurcular transit duration.
    See equation (3) from Seager and Malen-Ornelas, 2003ApJ...585.1038S.
    :param radius_1: R_star/a
    :param k: R_planet/R_star
    :param b: impact parameter = a.cos(i)/R_star
    :param P: orbital period (optional, default P=1)
    :returns: Total transit duration in the same units as P.
    """
    return  period*_math.asin(radius_1*_math.sqrt( ((1-k)**2-b**2) / (1-b**2*radius_1**2) ))/_math.pi


@_numba.njit
def stellar_density(radius_1, P, q=0, e = 0., w = _math.pi/2.):
    """ 
    Mean stellar density from scaled stellar radius.
    :param radius_1: radius of star in units of the semi-major axis, radius_1 = R_*/a
    :param P: orbital period in mean solar days
    :param q: mass ratio, m_2/m_1
    :returns: Mean stellar density in solar units
    # Eccentricity modification from https://arxiv.org/pdf/1505.02814.pdf   and  https://arxiv.org/pdf/1203.5537.pdf
    """
    fac =  (  ( ((1 - e**2))**1.5   ) / (( 1 + e*_math.sin(w) )**3) )  
    return (_rhostar/(P**2*(1+q)*radius_1**3))/fac 

@_numba.njit
def planet_logg_2(radius_2, period, K1, incl = _math.pi/2., e=0):
    """ 
    Companion surface gravity g = G.m_2/R_2**2 from P, K and radius_2
    
    Calculated using equation (4) from Southworth et al., MNRAS
    2007MNRAS.379L..11S. The
    :param radius_2: companion radius relative to the semi-major axis, radius_2 = R_2/a
    :param period: orbital period in mean solar days
    :param K1: semi-amplitude of star 1's orbit in km/s
    :param sini: sine of the orbital inclination
    :param e: orbital eccentrcity
    :returns: companion surface gravity in m.s-2
    """
    sini = _math.sin(incl)
    return 2*_math.pi*_math.sqrt(1-e**2)*K1*1e3/(period*mean_solar_day*radius_2**2*sini) 

@_numba.njit
def convert_width_and_period_to_radius_1(width, period):
    '''
    Assumes width and period are in days
    '''
    radius_1 =  _math.pi*width / period
    if _np.isnan(radius_1) or _np.isinf(radius_1) : return 0.2
    else : return radius_1

@_numba.njit
def convert_depth_to_k(depth):
    '''
    Assumed depth is in flux and positive from 0
    i.e. observed depth is 1 - depth
    '''
    depth = 1 - depth
    depth = -2.5*_math.log10(depth)
    depth =  _math.sqrt(depth)
    if _np.isnan(depth) or _np.isinf(depth) : return 0.1
    else : return depth

@_numba.njit
def equilbrium_temperature(Teff=5777, radius_1=0.1, albedo=0.5):
    return Teff*_math.sqrt(radius_1/2.)*(1 - albedo)**0.25

@_numba.njit
def insolation(luminosity, semi_major_axis):
    # Here luminosity is in solar units
    # semi_major_axis is in AU
    return luminosity / semi_major_axis**2


@_numba.njit
def scale_height(planet_temperature, planet_gravity, mu=2.3):
    return k_b*planet_temperature / (mu*u_kg*planet_gravity)

@_numba.njit
def transmission_signal(scale_height, k):
    return 2*scale_height*k**2

@_numba.njit
def estimate_RM_signal(k, b, vsini):
    # Estimate the anomalous RM effect signal 
    # Taken from https://arxiv.org/pdf/1001.2010.pdf
    return k**2*_math.sqrt(1 - b**2)*vsini

@_numba.njit
def estimate_period(duration ,density,b):
    # Eq1 from http://www.astro.yale.edu/jwang/LPPaper.pdf
    return 365.25*(duration*24/13)**3 * density * (1 - b**2)**(-1.5)

@_numba.njit
def JWST_transmission_metric_get_scale_factor(R2):
    # Get scale factor assuming R2 is in Earth radii
    if R2 < 1.5 : return 0.19
    elif R2 < 2.75 : return 1.26
    elif R2 < 4. : return 1.28
    elif R2 <  10 : return 1.15
    else : return -_np.inf

@_numba.njit
def JWST_transmission_metric(R1, R2,M2, Teq, Jmag ):
    # Eq1 from https://arxiv.org/pdf/1805.03671.pdf

    R2_ = R2* R_sun_earth
    M2_ = M2* M_sun_earth

    scale_factor = JWST_transmission_metric_get_scale_factor(R2_)
    return scale_factor*R2_**3*Teq*(10**(-0.2*Jmag)) / (M2_*R1**2)