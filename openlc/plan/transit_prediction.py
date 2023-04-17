from astroplan.constraints import months_observable
from astroplan import Observer, FixedTarget
from astropy.time import Time, TimeDelta
from astropy import units as u
from astroplan import (AltitudeConstraint, AirmassConstraint,
                       AtNightConstraint)
from astropy.coordinates import SkyCoord

from astroplan.constraints import time_grid_from_range
import numpy as np, matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from tqdm import tqdm 

__all__=['check_visibility_of_objects']

def months_observable(constraints, observer, targets,
                      time_grid_resolution=5*u.minute):
    """
    Determines which month the specified ``targets`` are observable for a
    specific ``observer``, given the supplied ``constriants``.

    Parameters
    ----------
    constraints : list or `~astroplan.constraints.Constraint`
        Observational constraint(s)

    observer : `~astroplan.Observer`
        The observer who has constraints ``constraints``

    targets : {list, `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`}
        Target or list of targets

    time_grid_resolution : `~astropy.units.Quantity` (optional)
        If ``time_range`` is specified, determine whether constraints are met
        between test times in ``time_range`` by checking constraint at
        linearly-spaced times separated by ``time_resolution``. Default is 0.5
        hours.

    Returns
    -------
    observable_months : list
        List of sets of unique integers representing each month that a target is
        observable, one set per target. These integers are 1-based so that
        January maps to 1, February maps to 2, etc.

    """
    # TODO: This method could be sped up a lot by dropping to the trigonometric
    # altitude calculations.
    if not hasattr(constraints, '__len__'):
        constraints = [constraints]

    # Calculate throughout the year of 2014 so as not to require forward
    # extrapolation off of the IERS tables
    time_range = Time(['2014-01-01', '2014-12-31'])
    times = time_grid_from_range(time_range, time_grid_resolution)

    # TODO: This method could be sped up a lot by dropping to the trigonometric
    # altitude calculations.

    applied_constraints = [constraint(observer, targets,
                                      times=times,
                                      grid_times_targets=True)
                           for constraint in constraints]
    print(applied_constraints[0][0].shape)
    constraint_arr = np.logical_and.reduce(applied_constraints)

    months_observable = []
    for target, observable in zip(targets, constraint_arr):
        s = set([t.datetime.month for t in times[observable]])
        months_observable.append(s)    

    return months_observable


def numberOfDays(y, m):
    leap = 0
    if y% 400 == 0:
        leap = 1
    elif y % 100 == 0:
        leap = 0
    elif y% 4 == 0:
        leap = 1
    if m==2:
        return 28 + leap
    list = [1,3,5,7,8,10,12]
    if m in list:
        return 31
    return 30


def months_observable(constraints, observer, targets,
                      time_grid_resolution=0.5*u.hour, year=2023):
    """
    Determines which month the specified ``targets`` are observable for a
    specific ``observer``, given the supplied ``constriants``.

    Parameters
    ----------
    constraints : list or `~astroplan.constraints.Constraint`
        Observational constraint(s)

    observer : `~astroplan.Observer`
        The observer who has constraints ``constraints``

    targets : {list, `~astropy.coordinates.SkyCoord`, `~astroplan.FixedTarget`}
        Target or list of targets

    time_grid_resolution : `~astropy.units.Quantity` (optional)
        If ``time_range`` is specified, determine whether constraints are met
        between test times in ``time_range`` by checking constraint at
        linearly-spaced times separated by ``time_resolution``. Default is 0.5
        hours.

    Returns
    -------
    observable_months : list
        List of sets of unique integers representing each month that a target is
        observable, one set per target. These integers are 1-based so that
        January maps to 1, February maps to 2, etc.

    """
    # TODO: This method could be sped up a lot by dropping to the trigonometric
    # altitude calculations.
    if not hasattr(constraints, '__len__'):
        constraints = [constraints]

    out = {}
    for i in tqdm(range(len(targets)), desc='Visibilty'): #target in targets:
        print()
        data = {}
        for month in range(1,13):
            time_start = Time('{:}-{:02}-01'.format(year, month))
            time_end = time_start + TimeDelta(numberOfDays(year, month), format='jd')
            time_range = Time([time_start.datetime.__str__()[:10], time_end.datetime.__str__()[:10]])
            times = time_grid_from_range(time_range, time_grid_resolution)

            applied_constraints = [constraint(observer, targets[i],
                                            times=times,
                                            grid_times_targets=True)
                                for constraint in constraints]
            constraint_arr = np.logical_and.reduce(applied_constraints)[0]
            s, count = np.unique([t.datetime.day for t in times[constraint_arr]], return_counts=True)

            count = (count*time_grid_resolution).to(u.minute)
            data[str(month)] = np.vstack((s,count.to_value())).astype(int)
        out[targets[i].name] = data   

    return out


def vidsualise_season_visibility(data, year='2023', min_minutes=0):
    # Now we need to do each object
    fig, ax = plt.subplots(1,1, figsize=(6,0.2*len(data.keys())))

    image = np.zeros((len(data.keys()), 365))

    i = 0
    for i in range(len(data.keys())) : 
        offset = 0
        for month in range(1,13):
            ndays_in_month = numberOfDays(int(year), month)
            for day in range(ndays_in_month+1):
                if day in data[list(data.keys())[i]][str(month)][0]:
                    idx = np.where(day==data[list(data.keys())[i]][str(month)][0])[0][0]
                    image[i,offset+day-1] = data[list(data.keys())[i]][str(month)][1][idx]/60
            offset += (ndays_in_month)

    # Apply the minimum
    image[image<(min_minutes/60)] = 0.
    time_lim = Time(['{:}-01-01 00:00'.format(year), '{:}-01-01 00:00'.format(int(year)+1)])
    ax.imshow(image, cmap='Greys', aspect='auto', extent= [time_lim.plot_date[0],time_lim.plot_date[1],  0,   len(data.keys())] )    

    cbar = ax.figure.colorbar(
            mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(min_minutes/60, 1440/2/60), cmap='Greys'),
            ax=ax, pad=0.01, extend='both', fraction=0.1, label='Hours visible')  

    ax.set(yticks=np.arange(len(data.keys()))+0.5, yticklabels=[i for i in data.keys()])

    ax.xaxis_date()
    date_format = mdates.DateFormatter('%m:%d')
    ax.xaxis.set_major_formatter(date_format)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout()                        
    return fig, ax



def check_visibility_of_objects(RA, Dec, names,location='Warwick', year=2023, time_grid_resolution=0.5*u.hour,
                constraints = [AltitudeConstraint(40*u.deg, 90*u.deg),AirmassConstraint(3), AtNightConstraint.twilight_civil()], min_minutes=0):
    # First, lets create the fixed targets from the skycoord
    targets = [FixedTarget(SkyCoord(RA[i]*u.deg, Dec[i]*u.deg, frame='icrs') , name = names[i]) for i in range(len(names))]

    # Define the observer 
    observer = Observer.at_site(location)

    # Gett the data
    data = months_observable(constraints, observer, targets,
                      time_grid_resolution=time_grid_resolution, year=year)

    fig,ax = vidsualise_season_visibility(data, year=str(year), min_minutes=min_minutes)

    return data, fig, ax 

# TEST
#N=1
#data, fig, ax  = check_visibility_of_objects(np.linspace(0,360,N), np.random.uniform(10,90,N), ['TIC-{:}'.format(i) for i in range(N)],location='Warwick')
#plt.show()