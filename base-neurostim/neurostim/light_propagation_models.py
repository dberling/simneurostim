import numpy as np

def fiber_Foutz2012(x, y, z, diameter_um, NA, spreading=True, scattering=True,
        scatter_coefficient=7.37, absorbance_coefficient=0.1249):
    """
    Optical fiber light source from Foutz et al 2012.
    NormalizedIntensity = 1 / (pi * (diameter_um/2)**2)
              * geom_loss * scattering_loss * gaussian_prof
    Extracted code from class Optrode in Foutz2012/classes.py and
    from file Foutz2012/functions.py.
    parameters:
    -----------
    x,y,z: float
        coordinates in um
    diameter_um: float
        diameter in um
    NA : float
        numerical aperture of the optical fiber
    ...
    scatter_coefficient: float
        scatter coeff in 1/mm, defaults to 7.37 from FOutz et al. 2012
    absorbance_coefficient: float
        absorbance coeff in 1/mm, defaults to 0.1249 from Foutz et al. 2012
    returns normalized intensity in W/cm2
    """
    # make z always positive to ensure correct light profile
    # whether positive or negative z axis are considered
    z = np.abs(z)
    from numpy import sqrt, arcsin

    optrode_radius = diameter_um / 2.0
    # parameters from Foutz et al. 2012:
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
        1 / (np.pi * (optrode_radius * 1e-4) ** 2) * Sx * Kx * gaussian(r, radius)
    )
