: mod file that allows control of conductance over time from python

UNITS {
    (mV) = (millivolt)
    (nA) = (nanoamp)
    (nS) = (nanosiemens)
}

NEURON { 
    SUFFIX g_chanrhod :Name of the Channel
    NONSPECIFIC_CURRENT	icat2
    RANGE gcat2    
    GLOBAL ecat
}

PARAMETER {
    ecat     = 0      (mV)     : Nagel 2003
}

ASSIGNED {  :calculated by the mechanism (computed by NEURON)
    v           (mV)
    icat2       (nA)
    gcat2       (nS) 
}

STATE { :state or independent variables
}

INITIAL {
    gcat2 = 0
}

BREAKPOINT {
    icat2 = gcat2 * 0.001 * (v-ecat) : input: nS and mV, hence need conversion factor 0.001 for nA
}
