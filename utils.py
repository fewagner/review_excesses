from math import erf
from typing import Union, Iterable
import numpy as np

# copied from https://github.com/fewagner/cait/blob/master/cait/limit/_yellin.py
# without using Numba

# @nb.njit
def I(m_N, speed_light, e_recoil, mu_N):
    """
    Integral of velocity distribution.
    """
    v_esc = 544.  # in km s^-1
    w = 220. * np.sqrt(3. / 2.)  # in km s^-1
    v_sun = 231.  # in km s^-1

    z = np.sqrt(3. / 2.) * v_esc / w
    eta = np.sqrt(3. / 2.) * v_sun / w
    norm = erf(z) - 2. / np.sqrt(np.pi) * z * np.exp(-(z ** 2))

    ret = 1. / (norm * eta) * np.sqrt(3. / (2. * np.pi * (w ** 2)))

    # ↓↓↓ from https://arxiv.org/pdf/hep-ph/9803295.pdf, eq. (A3)
    x_min = np.sqrt(3. * (speed_light ** 2) / 4.e12) * \
            np.sqrt(m_N * e_recoil / ((mu_N ** 2) * (w ** 2)))

    if x_min < z - eta:
        ret *= np.sqrt(np.pi) / 2. * (erf(x_min + eta) - erf(x_min - eta)) - 2. * eta * np.exp(-z ** 2)
    elif x_min < z + eta:
        ret *= np.sqrt(np.pi) / 2. * (erf(z) - erf(x_min - eta)) - (z + eta - x_min) * np.exp(-z ** 2)
    else:
        ret = 0.

    return ret



# @nb.njit
def F(e_recoil_: Union[int, float],
      m_N,
      a_nucleons):
    """
    Form factor.
    """
    if e_recoil_ <= 0.:
        return 1.

    # q is the transferred momentum, hbar * c = 197.326960
    q = np.sqrt(2. * m_N * e_recoil_) / 197.326960
    F_a = .52  # in fm
    F_s = .9  # in fm
    F_c = 1.23 * a_nucleons ** (1. / 3.) - .6  # in fm

    R_0 = np.sqrt((F_c ** 2) + 7. / 3. * (np.pi ** 2) * (F_a ** 2) - 5. * (F_s ** 2))

    # ret = spherical_jn(1, q * R_0) / (q * R_0)
    ret = q * R_0
    ret = np.sin(ret) / ret ** 2 - np.cos(ret) / ret
    ret /= (q * R_0)
    ret *= 3. * np.exp(-.5 * (q ** 2) * (F_s ** 2))

    return ret


# @nb.njit
def expected_interaction_rate(e_recoil: Union[int, float],
                              m_chi: Union[int, float],
                              a_nucleons: int,
                              ):
    """
    Yields the expected differential interaction rate between Dark Matter particles and a compound-nucleus.

    :param e_recoil: Given in keV.
    :param m_chi: Considered mass of the Dark Matter particle in GeV.
    :param a_nucleons: Number A of nucleons in the compound-nucleus.
    :return: dR/dE in kg^-1 d^-1 keV^-1.
    """
    # constants

    # speed of light in m s^-1
    speed_light = 299792458.
    # elementary charge in C
    e_charge = 1.60217662e-19
    # mass of the proton in GeV c^-2
    m_p = 0.9382720
    # mass of the nucleus
    m_N = m_p * a_nucleons
    # reduced mass of WIMP and proton
    mu_p = m_p * m_chi / (m_p + m_chi)
    # reduced mass of WIMP and nucleus
    mu_N = m_N * m_chi / (m_N + m_chi)
    # WIMP density on earth in GeV c^-2 cm^-3
    rho_chi = .3

    pre_factor = .5 * (speed_light ** 4) / (1.e12 * e_charge) * 1.e-40 * 86400. * (a_nucleons ** 2) / (m_chi * (mu_p ** 2))

    dN_per_dE = pre_factor  # * (a_nucleons ** 2)
    dN_per_dE *= rho_chi
    dN_per_dE *= (F(e_recoil, m_N, a_nucleons) ** 2)
    dN_per_dE *= I(m_N, speed_light, e_recoil, mu_N)

    return dN_per_dE
