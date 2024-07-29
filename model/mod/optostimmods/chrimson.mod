TITLE ChrimsonR Model according to Sabatier et al. (biorxiv) implemented by D.Berling

UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (nA) = (nanoamp)
    (mW) = (milliwatt)
}

NEURON { :public interface of the mechanism
    SUFFIX chrimson :Name of the Channel
    NONSPECIFIC_CURRENT	icat :Specific Channel for Na, Ca, K
    RANGE flux :photon flux on molecule
    RANGE channel_density :Channel density
    RANGE x, y, z :Location of that segment (of nsegs)
    GLOBAL source_flux :Optical Input
    GLOBAL h, c
    RANGE source_intensity, photon_energy
    RANGE wavelength
    GLOBAL gcat1, gcat2 : single channel conductance in state 1 / 2
    GLOBAL ecat : nernst potential for chanrhod
    RANGE Tx : light loss between source and segment
    GLOBAL tstimon, tstimoff
    GLOBAL O1toC1, O2toC2, O2toS, C2toC1, StoC1
    GLOABL PhoC1toO1, PhoC2toO2, PhoC1toC2, PhoC2toC1
}

PARAMETER {
    channel_density        = 1.3e10              (1/cm2) : variable : number of channels per cm2

    gcat1    = 50e-15    (mho)   : (Grossman 2011) 50 fS
    gcat2    = 250e-17   (mho)   : Figure 8 Nikolic 2009

    ecat     = 0      (mV)     : Nagel 2003

    Tx      = 1       (1)      : Default light loss between source and segment
    
    x = 0 (1) : spatial coords
    y = 0 (1)
    z = 0 (1)

    h = 6.6260693e-34        (m2 kg/s)  : planck's constant
    c = 299792458.0          (m/s)      : speed of light
    wavelength = 595e-9

}

ASSIGNED {  :calculated by the mechanism (computed by NEURON)
    v           (mV)
    icat        (mA/cm2)
    source_flux        (photons/ms) : flux of photons exiting optrode per millisecond, from ostim.mod
    flux               (photons/(ms) : number of photons reaching segment per area, from ostim.mod
    tstimon
    tstimoff
}

STATE { :state or independent variables
	O1 O2 C1 C2 S
}

INITIAL {
    flux = 0
    tstimon = 0
    tstimoff = 0

    : STATES
    C1 = 0.2 :Amount of channels at initial time
    C2 = 0
    O1 = 0
    O2 = 0
    S  = 0.8
}

BREAKPOINT {
    flux      = source_flux * Tx           : (photons/ms) * (1/cm2)
    
    icat = (O1 * gcat1 + O2 * gcat2) * channel_density * (v-ecat) : mA/cm2
	
    SOLVE states METHOD cnexp
    if (O1>1){O1=1}
    if (O1<0){O1=0}
    if (O2>1){O2=1}
    if (O2<0){O2=0}
    if (C1>1){C1=1}
    if (C1<0){C1=0}
    if (C2>1){C2=1}
    if (C2<0){C2=0}
    C1    = 1 - O1 - O2 - C2 - S
}

DERIVATIVE states {  
	O1' = - O1toC1 * O1                    + PhoC1toO1 * flux * C1
        O2' = - O2toC2 * O2                    + PhoC2toO2 * flux * C2            - O2toS * O2
        S'  = - StoC1  *  S + O2toS * O2
        :C1'= O1toC1 * O1 - PhoC1toO1 * flux * C1 - PhoC1toC2 * flux * C1 + C2toC1 * C2 + PhoC2toC1 * flux * C2 + StoC1 * S
        C2' =   O2toC2 * O2 - C2toC1 * C2      - PhoC2toC1 * flux * C2 + PhoC1toC2 * flux * C1 - PhoC2toO2 * flux * C2

}
