from matplotlib.pyplot import locator_params
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def cylindric(x, y, z, width, power, **kargs):
    """
    Cylindrical light in z direction.
    Intensity at source is power / (pi * (width/2)**2)
    parameters:
    -----------
    x,y,z: float
        coordinates in um
    width: float
        diameter of the cylinder in um
    power: float
        power of light in mW

    returns intensity
        intensity in mW/cm2
    """
    radius = width / 2.0
    return power / (np.pi * (radius * 1e-4) ** 2) * (np.sqrt(x ** 2 + y ** 2) < radius)

def yona_et_al2016(x, y, z, width, power, intensity_profile):
    """
    Optical fiber light source from Yona et al 2016.
    Intensity = power / (pi * (width/2)**2)
              * loss_calculated_from_beam_spread_funtion(BSF)
    
    parameters:
    -----------
    x,y,z: float
        coordinates in um
    width: float
        diameter in um
    power: float
        power of light in mW
    intensity_profile: dict
        "rhorho": 2D numpy array with radius coords
        "zz": 2D numpy array with z coords
        "I": relative loss simulated for specific parameters

    returns intensity
        intensity in mW/cm2
    """
    
    assert z<0, "z-coord got negative value but is defined only for positive"
    # model is defined to shine light on positive z
    z = -1 * z

    optrode_radius = width/2
    source_intensity = power / (np.pi * (optrode_radius * 1e-4) ** 2)
    # cylindrical coordinates:
    rho = np.sqrt(x**2 + y**2)
    
    # use simulated light profile data and associated
    # rho, z coordinates (rhorho,zz)
    Tx = griddata(
            (intensity_profile['rhorho'].flatten(), 
             intensity_profile['zz'].flatten()),
            intensity_profile['I'].flatten(),
            xi=(rho,z),
            method='linear'
        )
    return power * Tx


def soltan_et_al2017(x, y, z, width, power, intensity_profile):
    """
    Single LED light source from array described in Soltan et al 2017.
    Intensity = power * angle_profile * geometric_loss
    parameters:
    -----------
    x,y,z: float
        coordinates in um
    width: float
        unused (exists for consistency with other sources)
    power: float
        power of light in mW
    intensity_profile: list
        intensity vs angle profile for equidistant angle steps

    returns intensity
        intensity in mW/cm2
    """
    # convert radius to cm
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2) / 1e4
    n = len(intensity_profile)
    # convert to spherical coords and in radiants
    angle = np.arcsin(np.sqrt(x ** 2 + y ** 2) / np.sqrt(x ** 2 + y ** 2 + z ** 2))
    idx = np.int((n - 1) * angle / (np.pi / 2))
    # print('n',n, 'idx',idx,'x',x,'z',z)
    intensity = power * intensity_profile[idx] / r ** 2
    return intensity


def foutz_et_al2012(x, y, z, width, power, spreading=True, scattering=True, NA=0.37,
                    scatter_coefficient=7.37, absorbance_coefficient=0.1249, **kargs):
    """
    Optical fiber light source from Foutz et al 2012.
    Intensity = power / (pi * (width/2)**2)
              * geom_loss * scattering_loss * gaussian_prof
    Note: in the simulation it is used with power=1W so that any
    power can be multiplied to receive the actual intensity at specific power
    (because intensity is linear with power)
    Extracted code from class Optrode in Foutz2012/classes.py and
    from file Foutz2012/functions.py.
    parameters:
    -----------
    x,y,z: float
        coordinates in um
    width: float
        diameter in um
    power: float
        power of light in W
    NA : float
        numerical aperture of the optical fiber
    ...
    scatter_coefficient: float
        scatter coeff in 1/mm, defaults to 7.37 from FOutz et al. 2012
    absorbance_coefficient: float
        absorbance coeff in 1/mm, defaults to 0.1249 from Foutz et al. 2012
    returns intensity
        intensity in W/cm2
    """
    # make z always positive to ensure correct light profile
    # whether positive or negative z axis are considered
    z = np.abs(z)
    from numpy import sqrt, arcsin

    # parameters from Foutz et al. 2012:
    optrode_radius = width / 2.0
    n = 1.36  # index of refraction of gray matter
    theta_div = arcsin(NA / n)

    def gaussian(r, radius):
        """r is displacement from center
        95.4 % of light is within the radius (2 standard deviations)
        constant energy in distribution
        """
        from numpy import array, pi, sqrt, exp

        R = array(r) / array(radius)
        dist = (1 / sqrt(2 * pi)) * exp(-2 * R ** 2)
        # to do: initially in Foutz code here is dist/0.4
        # however we find integration over half-sphere equal 1 for dist/0.2
        return dist / 0.2

    def kubelka_munk(distance):
        """
        distance to center of optrode, approximates mean distance to all points along surface of optrode
        distance in um
        """
        from numpy import sqrt, sinh, cosh

        # K = 0.1248e-3 # 1/um
        # S = 7.37e-3   # 1/um
        K = absorbance_coefficient * 1e-3  # (1/um) # Range: (0.05233, 0.1975)
        S = scatter_coefficient * 1e-3  # (1/um) # Range: (6.679, 8.062)
        a = 1 + K / S  # unitless
        b = sqrt(a ** 2 - 1)  # unitless
        Tx = b / (
            a * sinh(b * S * distance) + b * cosh(b * S * distance)
        )  # distance in um - losses due to absorption and scattering through the tissue on top of losses due to beam quality?
        # Tx[distance<0]=0 # negative values set to zero
        Tx = Tx * np.array(distance>0).astype(int)  # negative values set to zero
        return Tx

    def apparent_radius(Z, R):
        """Find the apparent radius at a distance Z"""
        from numpy import tan

        return R + Z * tan(theta_div)

    def spread(Z):
        """irradiance loss due to spreading"""
        from numpy import sqrt

        rho = optrode_radius * sqrt(((n / NA) ** 2) - 1)
        return rho ** 2 / ((Z + rho) ** 2)

    def spread_david(Z):
        R_0 = optrode_radius
        R_Z = apparent_radius(Z, R_0)
        return (R_0 / R_Z) ** 2

    r = sqrt(x ** 2 + y ** 2)
    if scattering:  # kubelka-munk scattering
        Kx = kubelka_munk(sqrt(r ** 2 + z ** 2))
    else:
        Kx = 1
    if spreading:  # conservation of energy spreading
        #Sx = spread(z)
        Sx = spread_david(z)
        radius = apparent_radius(z, optrode_radius)
    else:
        Sx = 1
        radius = optrode_radius
    return (
        power / (np.pi * (optrode_radius * 1e-4) ** 2) * Sx * Kx * gaussian(r, radius)
    )


def foutz_NA0(x, y, z, width, power, spreading=True, scattering=True, **kargs):
    return foutz_et_al2012(x, y, z, width, power, spreading, scattering, NA=0)

def kubelka_LED(
    x,
    y,
    z,
    width,
    power,
    intensity_profile,
    distribution=True,
    scattering=True,
    **kargs
):
    """
    Close field approximation makes source non-singular in spreading part.

    LED light source spreading with Kubelka model scattering.
    Intensity = power / (pi * (width/2)**2)
              * LED_spreading * scattering_loss
    -----------
    x,y,z: float
        coordinates in um
    account for distance from LED source to the cortex in z coordinate (z starts from few um)
    width: float
        diameter in um
    power: float
        power of light in mW

    returns intensity
        intensity in mW/cm2
    """
    # make z always positive to ensure correct light profile
    z = np.abs(z)
    from numpy import sqrt

    LEDradius = width / 2.0
    # parameters from Foutz et al. 2012:
    n = 1.36  # index of refraction of gray matter
    absorbance_coefficient = 0.1249 * 1e-3  # (1/um) # Range: (0.05233, 0.1975)
    scatter_coefficient = 7.37 * 1e-3  # (1/um) # Range: (6.679, 8.062)

    def kubelka_munk(distance):
        """
        accounts for absorption and scattering losses in material
        distance in um
        """
        from numpy import sqrt, sinh, cosh

        K = absorbance_coefficient  # (1/um) # Range: (0.05233, 0.1975)
        S = scatter_coefficient  # (1/um) # Range: (6.679, 8.062)
        a = 1 + K / S  # unitless
        b = sqrt(a ** 2 - 1)  # unitless
        return b / (a * sinh(b * S * distance) + b * cosh(b * S * distance))

    def LEDdistr(r, distance):
        """units independent
        r = xx+yy
        distance = xx+yy+zz
        """
        dataLen = len(intensity_profile)

        """If one wants to approximate for bigger LED diameter, Intensity needs to be flat over LED and decrease to sides.
        Light goes to denser material => angle decreases => take data for n-times bigger angle"""
        angle = np.arcsin(r / distance) * n

        # convert to spherical coords, radians
        idx = np.int((dataLen - 1) * angle / (np.pi / 2))

        # Due to index of refraction modification, index might be longer than light_propagation_models data.
        # normalization comes from nonsingular source, it does not change if one accounts for LED diameter
        if idx > dataLen:
            return 0
        else:
            return (
                intensity_profile[idx] / 0.0891409
            )  # normalization constant comes from the change in index of refraction

    def spread(distance):
        """irradiance loss due to spreading from finite sized source"""
        return 1 / (2 * np.pi * distance ** 2)

    r = sqrt(x ** 2 + y ** 2)
    distance = sqrt(x ** 2 + y ** 2 + z ** 2)
    if distribution:
        Lx = LEDdistr(r, distance)
    else:
        Lx = 1
    if scattering:
        Kx = kubelka_munk(sqrt(r ** 2 + z ** 2))
    else:
        Kx = 1

    return power * spread(distance) * Lx * Kx


def kubelka_LED_closeField(
    x,
    y,
    z,
    width,
    power,
    intensity_profile,
    distribution=True,
    scattering=True,
    **kargs
):
    """
    Close field approximation makes source non-singular in spreading part.

    LED light source spreading with Kubelka model scattering.
    Intensity = power / (pi * (width/2)**2)
              * LED_spreading * scattering_loss
    -----------
    x,y,z: float
        coordinates in um
    account for distance from LED source to the cortex in z coordinate (z starts from few um)
    width: float
        diameter in um
    power: float
        power of light in mW

    returns intensity
        intensity in mW/cm2
    """
    # make z always positive to ensure correct light profile
    z = np.abs(z)
    from numpy import sqrt

    LEDradius = width / 2.0
    # parameters from Foutz et al. 2012:
    n = 1.36  # index of refraction of gray matter
    absorbance_coefficient = 0.1249 * 1e-3  # (1/um) # Range: (0.05233, 0.1975)
    scatter_coefficient = 7.37 * 1e-3  # (1/um) # Range: (6.679, 8.062)

    def kubelka_munk(distance):
        """
        accounts for absorption and scattering losses in material
        distance in um
        """
        from numpy import sqrt, sinh, cosh

        K = absorbance_coefficient  # (1/um) # Range: (0.05233, 0.1975)
        S = scatter_coefficient  # (1/um) # Range: (6.679, 8.062)
        a = 1 + K / S  # unitless
        b = sqrt(a ** 2 - 1)  # unitless

        def Tx(distance):
            return b / (a * sinh(b * S * distance) + b * cosh(b * S * distance))

        if r < LEDradius:
            return Tx(z) * float(distance > 0)
        else:
            return Tx(sqrt((r - LEDradius) ** 2 + z ** 2))

    def LEDdistr(r, distance):
        """units independent
        r = xx+yy
        distance = xx+yy+zz
        """
        dataLen = len(intensity_profile)

        """If one wants to approximate for bigger LED diameter, Intensity needs to be flat over LED and decrease to sides.
        Light goes to denser material => angle decreases => take data for n-times bigger angle"""
        angle = np.arcsin(r / distance) * n
        """or for more advanced approximation, commented code below can be used. This would have to be normalized according to LED width
        if r<LEDradius:
            angle = 0
        else:
            angle = np.arcsin((r-LEDradius) / distance) * n
        """

        # convert to spherical coords, radians
        idx = np.int((dataLen - 1) * angle / (np.pi / 2))

        # Due to index of refraction modification, index might be longer than light_propagation_models data.
        # normalization comes from nonsingular source, it does not change if one accounts for LED diameter
        if idx > dataLen:
            return 0
        else:
            return (
                intensity_profile[idx] / 0.0891409
            )  # normalization constant comes from the change in index of refraction

    def spread(r, z):
        """irradiance loss due to spreading from finite sized source
        substitutes 1/r^2 for point source
        """
        from numpy import pi, sqrt

        # normalization of spread comes from weird shape - disc+torus/4 is the surface
        def S(Z):
            return pi * LEDradius ** 2 + pi ** 2 * Z * (LEDradius + Z / 2.0)

        if r < LEDradius:
            return 1 / S(z) / 1.2732395
        else:
            return 1 / S(sqrt((r - LEDradius) ** 2 + z ** 2)) / 1.2732395

    r = sqrt(x ** 2 + y ** 2)
    distance = sqrt(x ** 2 + y ** 2 + z ** 2)
    if distribution:
        Lx = LEDdistr(r, distance)
    else:
        Lx = 1
    if scattering:
        Kx = kubelka_munk(sqrt(r ** 2 + z ** 2))
    else:
        Kx = 1

    return power * spread(r, z) * Lx * Kx

def gaussian(x, mu, sig):
    return (2*np.pi)**(-0.5) / abs(sig) * np.exp(- (x-mu)**2 / (2 * sig**2))

def usp_model(power, r, z, r_sig, z_sig):
    """ 
    ultrasound emitter model based on experimental data from Cadoni et al. 2021 Fig. 1

    Models ultrasound profile which is elliptic as product of gaussians.
    r_sig = 0.5, z_sig = 10 gives approximate profile for 2.25 MHz
    r_sig = 0.1, z_sig = 2 gives approximate profile for 15 MHz
    """
    return power * gaussian(r, mu=0, sig=r_sig) * gaussian(z,mu=0,sig=z_sig)

def usp_model_wrap(x,y,z, width, power, **kargs):
    """
    wrapper for usp_model to follow convention for light_propagation_model
    x,y,z in um need to be converted to mm
    """
    x*=0.001
    y*=0.001
    z*=0.001
    if width == 2250:
        # set 2.25 MHz settings
        r_sig = 0.5
        z_sig = 10
    elif width == 15000:
        # set 15 MHz settings
        r_sig = 0.1
        z_sig = 2
    return usp_model(power=power, r=np.sqrt(x**2+y**2), z=z, r_sig=r_sig, z_sig=z_sig)

def usp_real_model(x,y,z,power,interpolation_object,max_power_interpolated_data, **kargs):
    """
    x,y,z: coordinates in um
    power: pressure in MPa
    interpolation_object: e.g. scipy.interpolate.RegularGridInterpolator, use um for xy and z
                          usage: instantiated object takes 2 arguments: z, xy
                                 and returns power in MPa
    max_power_interpolated_data: maximal power reached throughout cortical space in 
                                 interpolated data, is used to normalize.
    """
    return power / max_power_interpolated_data * interpolation_object(z, np.sqrt(x**2 + y**2))
