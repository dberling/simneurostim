from optostim.light_classes import LightSource
from optostim.light_propagation_models import (
    foutz_et_al2012,
    soltan_et_al2017,
    kubelka_LED,
    kubelka_LED_closeField,
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, nquad

model = "soltan_et_al2017"
intensity_angle_profile = LightSource(model, (0, 0, 0), 0).intensity_angle_profile

distributionVal = True
scatteringVal = True
LEDwidth = 45

"""integration using scipy
"""


def Ifoutz(t, p, r):
    return (
        foutz_et_al2012(
            r * np.cos(p) * np.sin(t),
            r * np.sin(p) * np.sin(t),
            -r * np.cos(t),
            500,
            1,
            distributionVal,
            scatteringVal,
        )
        * np.sin(t)
        * r ** 2
    )


def Isoltan(t, p, r):
    return (
        soltan_et_al2017(
            r * np.cos(p) * np.sin(t),
            r * np.sin(p) * np.sin(t),
            -r * np.cos(t),
            0,
            1,
            intensity_angle_profile,
        )
        * np.sin(t)
        * r ** 2
    )


def ILED(t, p, r):
    return (
        kubelka_LED(
            r * np.cos(p) * np.sin(t),
            r * np.sin(p) * np.sin(t),
            -r * np.cos(t),
            LEDwidth,
            1,
            intensity_angle_profile,
            distributionVal,
            scatteringVal,
        )
        * np.sin(t)
        * r ** 2
    )


def ILED_closeField(t, p, r):
    return (
        kubelka_LED_closeField(
            r * np.cos(p) * np.sin(t),
            r * np.sin(p) * np.sin(t),
            -r * np.cos(t),
            LEDwidth,
            1,
            intensity_angle_profile,
            distributionVal,
            scatteringVal,
        )
        * np.sin(t)
        * r ** 2
    )


def sphere_integrate(Imodel, r):
    def I(t, p):
        return Imodel(t, p, r)

    return nquad(I, [[0, np.pi / 2.0], [0, 2 * np.pi]])


def I(t, p):
    return ILED(t, p, 1)


nquad(I, [[0, np.pi / 2.0], [0, 2 * np.pi]])

sphere_integrate(ILED, 1000000000)
sphere_integrate(ILED_closeField, 1000000000)


"""kubelka_LED_closeField"""
n = 256
xlim = 100.0
x = np.linspace(-xlim, xlim, n)
z = np.linspace(0.0, xlim, n)
X, Z = np.meshgrid(x, z)

Ixz = [
    [
        kubelka_LED_closeField(
            xx,
            0,
            zz,
            LEDwidth,
            1,
            intensity_angle_profile,
            distributionVal,
            scatteringVal,
        )
        for xx in x
    ]
    for zz in z
]

plt.pcolormesh(X, Z, np.log(Ixz))
plt.xlabel("y /μm")
plt.ylabel("z /μm")
plt.show()

# sphere integration
rf = np.linspace(1, 1000, 100)
Irf = [sphere_integrate(ILED_closeField, rr)[0] for rr in rf]
plt.plot(rf, Irf)
plt.xlabel("r /μm")
plt.ylabel("Power flowing through half-sphere /mW")
plt.show()


"""kubelka_LED"""
n = 256
xlim = 100.0
x = np.linspace(-xlim, xlim, n)
z = np.linspace(0.0, xlim, n)
X, Z = np.meshgrid(x, z)

Ixz = [
    [
        kubelka_LED(
            xx,
            0,
            zz,
            LEDwidth,
            1,
            intensity_angle_profile,
            distributionVal,
            scatteringVal,
        )
        for xx in x
    ]
    for zz in z
]

plt.pcolormesh(X, Z, np.log(Ixz))
plt.xlabel("y /μm")
plt.ylabel("z /μm")
plt.show()

# sphere integration

rf = np.linspace(1, 1000, 10)
Irf = [sphere_integrate(ILED, rr)[0] for rr in rf]
plt.plot(rf, Irf)
plt.xlabel("r /μm")
plt.ylabel("Power flowing through half-sphere /mW")
plt.show()


"""
foutz_et_al2012 contains absorption and corresponds to soltan_et_al2017 model around r=200 μm and r=600 μm
light intensity along z coordinate is just decreasing
"""
z = np.linspace(0.01, 10000, 500)
Iz = [foutz_et_al2012(0, 0, zz, 500, 1, True, True) / 1e8 for zz in z]
plt.plot(z, Iz)
plt.xlabel("z /μm")
plt.ylabel("I /mW.cm^-2")
plt.show()
