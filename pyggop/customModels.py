# Custom model

import collections
import glob

import numpy

from astromodels.functions.function import *
from astromodels import ModelAssertionViolation
from astromodels.functions.functions import Band

import astropy.units as astropy_units

from pyggop import fast_flux_computation


class BandPP(Function1D):
    r"""
    description :

        Band model affected by pair production opacity

    latex : $  $

    parameters :

        K :

            desc : Differential flux at 100 keV
            initial value : 1e-4

        alpha :

            desc : low-energy photon index
            initial value : -1.0
            min : -1.5
            max : 0

        xp :

            desc : peak energy in the nuFnu spectrum
            initial value : 350

        beta :

            desc : high-energy photon index
            initial value : -2.0
            min : -3.0
            max : -1.5

        xc :

            desc : cutoff energy
            initial value : 3e4

        n :

            desc : delta photon index after the cutoff
            initial value : 1.0
            min : 0.1
            max : 10.0

        piv :

            desc : pivot energy
            initial value : 100.0
            fix : yes
    """

    __metaclass__ = FunctionMeta

    def _set_units(self, x_unit, y_unit):

        # The normalization has the same units as y
        self.K.unit = y_unit

        # The break point has always the same dimension as the x variable
        self.xp.unit = x_unit
        self.xc.unit = x_unit

        self.piv.unit = x_unit

        # alpha and beta are dimensionless
        self.alpha.unit = astropy_units.dimensionless_unscaled
        self.beta.unit = astropy_units.dimensionless_unscaled
        self.n.unit = astropy_units.dimensionless_unscaled

    def evaluate(self, x, K, alpha, xp, beta, xc, n, piv):

        E0 = xp / (2 + alpha)

        if (alpha < beta):
            raise ModelAssertionViolation("Alpha cannot be less than beta")

        out = np.where(x < (alpha - beta) * E0,
                       K * numpy.power(x / piv, alpha) * numpy.exp(-x / E0),
                       K * numpy.power((alpha - beta) * E0 / piv, alpha - beta) * numpy.exp(beta - alpha) *
                       numpy.power(x / piv, beta))

        # This fixes nan(s) and inf values, converting them respectively to zeros and large numbers
        out = numpy.nan_to_num(out)

        # Multiply by the Granot factor
        deltaBeta = (-1 - beta) * (2 - beta) / (1 - beta)

        granot = numpy.power(1 + numpy.power(x / xc, n * deltaBeta), -1 / n)

        out = out * granot

        return out


class ModifiedBand(Function1D):
    r"""
        description :

            A Band model with a power law break at high energy

        latex : $  $

        parameters :

            K :

                desc : Differential flux at 100 keV
                initial value : 1e-4

            alpha :

                desc : low-energy photon index
                initial value : -1.0
                min : -1.5
                max : 0

            xp :

                desc : peak energy in the nuFnu spectrum
                initial value : 350

            beta :

                desc : high-energy photon index
                initial value : -2.0
                min : -3.0
                max : -1.5

            xc :

                desc : cutoff energy
                initial value : 3e4

            delta :

                desc : delta photon index after the cutoff
                initial value : 0.5
                min : 0
                max : 5

            piv :

                desc : pivot energy
                initial value : 100.0
                fix : yes
        """

    __metaclass__ = FunctionMeta

    def _set_units(self, x_unit, y_unit):
        # The normalization has the same units as y
        self.K.unit = y_unit

        # The break point has always the same dimension as the x variable
        self.xp.unit = x_unit
        self.xc.unit = x_unit

        self.piv.unit = x_unit

        # alpha and beta are dimensionless
        self.alpha.unit = astropy_units.dimensionless_unscaled
        self.beta.unit = astropy_units.dimensionless_unscaled
        self.delta.unit = astropy_units.dimensionless_unscaled

    def evaluate(self, x, K, alpha, xp, beta, xc, delta, piv):

        E0 = xp / (2 + alpha)

        gamma = beta - delta

        if alpha < beta or xc < E0:

            raise ModelAssertionViolation("Alpha cannot be less than beta and xc cannot be less than E0")

        out1 = np.where(x < (alpha - beta) * E0,
                        K * numpy.power(x / piv, alpha) * numpy.exp(-x / E0),
                        K * numpy.power((alpha - beta) * E0 / piv, alpha - beta) * numpy.exp(beta - alpha) *
                        numpy.power(x / piv, beta))

        out2 = np.where(x < xc,
                        out1,
                        K * numpy.power((alpha - beta) * E0 / piv, alpha - beta) * numpy.exp(
                            beta - alpha) * numpy.power(
                            xc / piv, beta) * numpy.power(xc / piv, -gamma) * numpy.power(x / piv, gamma)
                        )

        # This fixes nan(s) and inf values, converting them respectively to zeros and large numbers
        out = numpy.nan_to_num(out2)

        return out


# Yoni's templates
import scipy.interpolate


# class PairProduction(object):
#     def __init__(self, templateFile, drOnR0):
#
#         oldStyle = False
#         with open(templateFile) as f:
#
#             line = f.readline()
#
#             if line[0] != '#':
#                 oldStyle = True
#
#         self.drOnR0 = float(drOnR0)
#
#         self.Ec = 10000  # init value
#
#         if oldStyle:
#
#             thisData = numpy.genfromtxt(templateFile,
#                                         delimiter=',')
#
#             self._e = numpy.power(10, thisData[:, 0])
#             self._nuFnu = numpy.power(10, thisData[:, 1])
#
#         else:
#
#             # New template style (pyggop)
#
#             thisData = numpy.genfromtxt(templateFile,
#                                         delimiter=' ',
#                                         comments='#')
#
#             self._e = thisData[:, 0]
#             self._nuFnu = thisData[:, 1]
#
#             # Replace zero values with a floor value
#             idx = self._nuFnu < 1e-20
#             self._nuFnu[idx] = 1e-20
#
#         # Make sure they are sorted
#         idx = self._e.argsort()
#         self._e = self._e[idx]
#         self._nuFnu = self._nuFnu[idx]
#
#         # Normalize the nuFnu to 1
#         self._nuFnu = self._nuFnu / self._nuFnu.max()
#
#         self._setInterpolant()
#
#     def _setInterpolant(self):
#
#         self.interpolant = scipy.interpolate.UnivariateSpline(numpy.log10(self._e * self.Ec),
#                                                               numpy.log10(self._nuFnu), k=1,
#                                                               s=0, ext=3)
#
#     def __call__(self, energies):
#         return numpy.power(10, self.interpolant(numpy.log10(energies)))
#
#     def set_cutoff_energy(self, Ec):
#         # print("Cutoff energy is now %s" %(Ec))
#         self.Ec = float(Ec)
#         self._setInterpolant()

_template_dir = os.getcwd()

def set_template_directory(dir):

    global _template_dir

    _template_dir = dir


class MyInterpolator(object):

    def __init__(self, templates_dir=None):

        self.interpolate = True

        # Make the interpolators

        if templates_dir is None:

            templates_dir = _template_dir

        templates = glob.glob(os.path.join(templates_dir,"flux_*.dat"))

        if len(templates)==0:

            warnings.warn("No template found in %s. Set the template directory." % templates_dir)

        else:
            # Get the energy grid from the first interpolator
            data = numpy.genfromtxt(templates[0], delimiter=' ', comments='#')

            # A-dimensional energy

            self.eneGrid = data[:, 0]

            # Read all the templates
            points = []
            values = []

            # Pre-computed templates
            self.cached = {}

            # Keep track of betas and DRbars
            self.betas = []
            self.DRbars = []

            for dat in templates:
                tokens = dat.split("_")
                beta = float(tokens[2].replace("a", ""))
                DRbar = float(tokens[4].replace("DR", "").replace(".dat", ""))

                points.append([beta, DRbar])

                data = numpy.genfromtxt(dat, delimiter=' ', comments='#')

                # Flux

                fl = data[:, 1]

                # Store the logarithm of the flux

                values.append(numpy.log10(fl))

                self.cached[(beta, DRbar)] = numpy.log10(fl)

                self.betas.append(beta)
                self.DRbars.append(DRbar)

            points = numpy.array(points)
            values = numpy.array(values)

            # Sort them first by index then by dr
            idx = numpy.lexsort((points[:, 1], points[:, 0]))
            points = points[idx]
            values = values[idx]

            # Now build one interpolator for each energy
            self.interpolators = []

            for i in range(self.eneGrid.shape[0]):

                this = scipy.interpolate.LinearNDInterpolator(points, values[:, i])

                self.interpolators.append(this)

    def set_interpolation_off(self):

        self.interpolate = False

    def get_template(self, beta, DRbar):

        if (beta, DRbar) in self.cached:

            return self.cached[(beta, DRbar)]

        if self.interpolate and DRbar <= max(self.DRbars) and DRbar >= min(self.DRbars):

            values = map(lambda interp: interp([beta, DRbar])[0], self.interpolators)

            # Logarithm of the flux

            return numpy.array(values)

        else:

            _, eps, f_time_int = fast_flux_computation.go(0, 0, beta, DRbar, 1.0, 1.0, False)

            # Logarithm of the flux

            return numpy.array(numpy.log10(f_time_int))


class BandPPTemplate(Function1D):
    r"""
        description :

            A Band model multiplied by a pair-production opacity computed by interpolating templates from the pyggop
            code

        latex : $  $

        parameters :

            K :

                desc : Differential flux at 100 keV
                initial value : 1e-4

            alpha :

                desc : low-energy photon index
                initial value : -1.0
                min : -1.5
                max : 0.0

            xp :

                desc : peak energy in the nuFnu spectrum
                initial value : 350
                min : 1
                max : 1e5

            beta :

                desc : high-energy photon index
                initial value : -2.0
                min : -2.3
                max : -1.7

            xc :

                desc : cutoff energy
                initial value : 3e4

            DRbar :

                desc : drbar
                initial value : 1
                min : 0.1
                max : 130

            piv :

                desc : pivot energy
                initial value : 100.0
                fix : yes
        """

    __metaclass__ = FunctionMeta

    def _setup(self):

        self.interpolator = MyInterpolator()

        self.cache = {}

        self._band_model = Band()

    def set_templates_dir(self, directory):

        self.interpolator = MyInterpolator(directory)

        self.cache = {}

    def set_template(self, templateInstance):

        self.templateInstance = templateInstance

    def _set_units(self, x_unit, y_unit):

        # The normalization has the same units as y
        self.K.unit = y_unit

        # The break point has always the same dimension as the x variable
        self.xp.unit = x_unit
        self.xc.unit = x_unit

        self.piv.unit = x_unit

        # alpha and beta are dimensionless
        self.alpha.unit = astropy_units.dimensionless_unscaled
        self.beta.unit = astropy_units.dimensionless_unscaled
        self.DRbar.unit = astropy_units.dimensionless_unscaled

    def evaluate(self, x, K, alpha, xp, beta, xc, DRbar, piv):

        if alpha < beta:
            raise ModelAssertionViolation("Alpha cannot be less than beta")

        #if xc <= xp:
        #    raise ModelAssertionViolation("Ec cannot be less than Ep")

        out = self._band_model.evaluate(x, K, alpha, xp, beta, piv)

        key = ("%.5f, %.4g" % (beta, DRbar))

        if key in self.cache.keys():

            # Logarithm of the flux

            template = numpy.array(self.cache[key], copy=True)

        else:

            # Logarithm of the flux

            template = self.interpolator.get_template(beta * (-1), DRbar)

            self.cache[key] = numpy.array(template, copy=True)

        pass

        # Energy grid in "observed energy"

        ee = self.interpolator.eneGrid * xc

        # Interpolate in the log space

        interpolation = numpy.interp(numpy.log10(x), numpy.log10(ee), template)

        # Go back to the flux space

        cc = numpy.power(10, interpolation)

        # Now go to photon flux
        cc = cc / np.power(x, -beta)

        # Correct the extremes in case the template does not cover them

        #idx = (x / xc < self.interpolator.eneGrid.min())
        #cc[idx] = cc.max()

        idx = (x / xc > self.interpolator.eneGrid.max())
        cc[idx] = 1e-35

        # Match the Band spectrum and the template at E0
        E0 = xp / (2 + alpha)
        Ec = (alpha - beta) * E0

        # Diff. flux of the Band spectrum at Ec

        flux_at_Ec = self._band_model.evaluate(np.array(Ec, ndmin=1), K, alpha, xp, beta, piv)

        # Template at Ec
        template_at_Ec = pow(10, numpy.interp(numpy.log10(Ec), numpy.log10(ee), template)) / pow(Ec, -beta)
        
        idx = (x >= Ec)

        # Renorm factor
        renorm = flux_at_Ec / template_at_Ec

        cc = renorm * cc

        # Now join the Band spectrum and the template

        out[idx] = cc[idx]

        # This should be np.power(x, -beta) * out * cc / np.power(x, -beta), which of course simplify to:

        #out = out * cc

        # This fixes nan(s) and inf values, converting them respectively to zeros and large numbers
        out = numpy.nan_to_num(out)

        return out

