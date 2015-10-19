#Custom model

from threeML.models.spectralmodel import SpectralModel
from threeML.models.Parameter import Parameter
from threeML.exceptions import CustomExceptions
import collections
import numpy
import glob

from pyggop import fast_flux_computation

class BandPP(SpectralModel):
    
    def setup(self):
        
        self.functionName       = "Band model with pair production opacity"
        self.formula            = r'''
        \[f(E) = \left\{ \begin{eqnarray}
        K \left(\frac{E}{100 \mbox{ keV}}\right)^{\alpha} & \exp{\left(\frac{-E}{E_{c}}\right)} & \mbox{ if } E < (\alpha-\beta)E_{c} \\
        K \left[ (\alpha-\beta)\frac{E_{c}}{100}\right]^{\alpha-\beta}\left(\frac{E}{100 \mbox{ keV}}\right)^{\beta} & \exp{(\beta-\alpha)} & \mbox{ if } E \ge (\alpha-\beta)E_{c}
        \end{eqnarray}
        \right.
        \]
        '''

        self.parameters         = collections.OrderedDict()
        self.parameters['alpha'] = Parameter('alpha',-1.0,-5,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['beta']  = Parameter('beta',-2.0,-10,0,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['E0']    = Parameter('E0',500,10,1e5,10,fixed=False,nuisance=False,dataset=None,unit='keV')
        self.parameters['K']     = Parameter('K',1,1e-4,1e3,0.1,fixed=False,nuisance=False,dataset=None,
                                             normalization=True)
        #[1+(E/E_c)^{n\Delta\beta}]^{-1/n}
        self.parameters['Ec']    = Parameter('Ec',3e4,1e3,1e6,10,fixed=False,nuisance=False,dataset=None,unit='keV')
        self.parameters['n']     = Parameter('n',1,0.1,3,0.1,fixed=False,nuisance=False,dataset=None)

        
        def integral(e1,e2):
            return self((e1+e2)/2.0)*(e2-e1)
        self.integral            = integral
    
  
  
    def __call__(self,e):
        #The input e can be either a scalar or an array
        #The following will generate a wrapper which will
        #allow to treat them in exactly the same way,
        #as they both were arrays, while keeping the output
        #in line with the input: if e is a scalar, the output
        #will be a scalar; if e is an array, the output will be an array
        energies                 = numpy.array(e,ndmin=1,copy=False)
        alpha                    = self.parameters['alpha'].value
        beta                     = self.parameters['beta'].value
        E0                       = self.parameters['E0'].value
        K                        = self.parameters['K'].value
        Ec                       = self.parameters['Ec'].value
        n                        = self.parameters['n'].value
        
        if(alpha < beta):
          raise CustomExceptions.ModelAssertionViolation("Alpha cannot be less than beta")
        
        out                      = numpy.zeros(energies.flatten().shape[0])
        idx                      = (energies < (alpha-beta)*E0)
        nidx                     = ~idx
        out[idx]                 = numpy.maximum(K*numpy.power(energies[idx]/100.0,alpha)*numpy.exp(-energies[idx]/E0),1e-30)
        out[nidx]                = numpy.maximum(K*numpy.power((alpha-beta)*E0/100.0,alpha-beta)*numpy.exp(beta-alpha)*numpy.power(energies[nidx]/100.0,beta),1e-30)
    
        #This fixes nan(s) and inf values, converting them respectively to zeros and large numbers
        out                      = numpy.nan_to_num(out)
        
        #Multiply by the Granot factor
        deltaBeta                = (-1 - beta) * (2 - beta) / (1 - beta)
        #print("db = %.2f" % deltaBeta)
        granot                   = numpy.power(numpy.power(1 + e/Ec, n * deltaBeta), -1/n)
        out                      = out * granot
        
        if(out.shape[0]==1):
            return out[0]
        else:
            return out
 


class ModifiedBand(SpectralModel):
    def setup(self):
        self.functionName       = "Band model with power law decay"
        self.formula            = r'''
        $f_{Bbkpo}(E) =$ \begin{cases}
K E^{\alpha} \exp{\left(\frac{-E}{E_{0}}\right)} & E < (\alpha-\beta)E_{0} \\ 
K \left[ (\alpha-\beta)E_{0}\right]^{\alpha-\beta} \exp{(\beta-\alpha)} E^{\beta} & (\alpha-\beta)E_{0} \le E < E_{c}\\
K \left[ (\alpha-\beta)E_{0}\right]^{\alpha-\beta} \exp{(\beta-\alpha)} E_{c}^{\beta-\gamma} E^{\gamma} & E \ge E_{c}
\end{cases}
        '''

        self.parameters         = collections.OrderedDict()
        self.parameters['alpha'] = Parameter('alpha',-1.0,-5,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['beta']  = Parameter('beta',-2.0,-10,0,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['E0']    = Parameter('E0',500,10,1e5,10,fixed=False,nuisance=False,dataset=None,unit='keV')
        self.parameters['K']     = Parameter('K',1,1e-4,1e3,0.1,fixed=False,nuisance=False,dataset=None,normalization=True)
        self.parameters['break'] = Parameter('break',3e4,10,1e7,1e3,fixed=False,nuisance=False,dataset=None,unit='keV')
        self.parameters['gamma'] = Parameter('gamma',-4.0,-10,0,0.1,fixed=False,nuisance=False,dataset=None)
        
        def integral(e1,e2):
            return self((e1+e2)/2.0)*(e2-e1)
        self.integral            = integral
    
  
  
    def __call__(self,e):
        #The input e can be either a scalar or an array
        #The following will generate a wrapper which will
        #allow to treat them in exactly the same way,
        #as they both were arrays, while keeping the output
        #in line with the input: if e is a scalar, the output
        #will be a scalar; if e is an array, the output will be an array
        energies                 = numpy.array(e,ndmin=1,copy=False)
        alpha                    = self.parameters['alpha'].value
        beta                     = self.parameters['beta'].value
        E0                       = self.parameters['E0'].value
        K                        = self.parameters['K'].value
        E1                       = self.parameters['break'].value
        gamma                    = self.parameters['gamma'].value
        
        if(alpha < beta):
          raise CustomExceptions.ModelAssertionViolation("Alpha cannot be less than beta")
        
        out                      = numpy.zeros(energies.flatten().shape[0])
        idx1                     = (energies < (alpha-beta)*E0)
        idx2                     = ~idx1 & (energies < E1)
        idx3                     = ~idx1 & ~idx2
        out[idx1]                = numpy.maximum(K*numpy.power(energies[idx1]/100.0,alpha)*numpy.exp(-energies[idx1]/E0),1e-30)
        out[idx2]                = numpy.maximum(K*numpy.power((alpha-beta)*E0/100.0,alpha-beta)*numpy.exp(beta-alpha)*numpy.power(energies[idx2]/100.0,beta),1e-30)
        out[idx3]                = numpy.maximum(K*numpy.power((alpha-beta)*E0/100.0,alpha-beta)*numpy.exp(beta-alpha)*numpy.power(E1/100.0,beta)*numpy.power(E1/100.0,-gamma)*numpy.power(energies[idx3]/100.0,gamma),1e-30)
        
        #This fixes nan(s) and inf values, converting them respectively to zeros and large numbers
        out                      = numpy.nan_to_num(out)
        if(out.shape[0]==1):
            return out[0]
        else:
            return out


#Yoni's templates
import scipy.interpolate

class PairProduction(object):
    
    def __init__(self,templateFile,drOnR0):
        
        oldStyle = False
        with open(templateFile) as f:
            
            line = f.readline()
            
            if line[0]!='#':
                 
                 oldStyle = True
        
        self.drOnR0 = float(drOnR0)
        
        self.Ec = 10000 #init value
        
        if oldStyle:
            
            thisData = numpy.genfromtxt(templateFile,
                                        delimiter=',')
            
            self._e = numpy.power(10,thisData[:,0])
            self._nuFnu = numpy.power(10, thisData[:,1])
        
        else:
            
            #New template style (pyggop)                        
            
            thisData = numpy.genfromtxt(templateFile,
                                        delimiter=' ',
                                        comments='#')

            self._e = thisData[:,0]
            self._nuFnu = thisData[:,1]
            
            #Replace zero values with a floor value
            idx = self._nuFnu < 1e-20
            self._nuFnu[idx] = 1e-20
            
        
        #Make sure they are sorted
        idx = self._e.argsort()
        self._e = self._e[idx]
        self._nuFnu = self._nuFnu[idx]
        
        #Normalize the nuFnu to 1
        self._nuFnu = self._nuFnu / self._nuFnu.max()
        
        self._setInterpolant()
    
    def _setInterpolant(self):
        
        self.interpolant = scipy.interpolate.UnivariateSpline(numpy.log10(self._e * self.Ec), 
                                                                          numpy.log10(self._nuFnu), k=1,
                                                                          s=0, ext=3)
    
    def __call__(self, energies):
        return numpy.power(10,self.interpolant(numpy.log10(energies)))
    
    def setCutoffEnergy(self, Ec):
        #print("Cutoff energy is now %s" %(Ec))
        self.Ec = float(Ec)
        self._setInterpolant()


class MyInterpolator( object ):
    
    def __init__( self ):
        
        #Make the interpolators
        
        templates = glob.glob("flux_*.dat")

        #Get the energy grid from the first interpolator
        data = numpy.genfromtxt( templates[0], delimiter=' ', comments='#')

        self.eneGrid = data[:,0]

        #Read all the templates
        points = []
        values = []

        for dat in templates:
            
            tokens = dat.split("_")
            beta = float(tokens[2].replace("a",""))
            DRbar = float(tokens[4].replace("DR","").replace(".dat",""))
            
            points.append( [beta, DRbar] )
            
            data = numpy.genfromtxt( dat, delimiter=' ', comments='#')
        
            fl = data[:,1]
            #idx = fl < 1e-30
            #fl[idx] = 1e-30
    
            values.append( numpy.log10(fl) )
        
        points = numpy.array(points)
        values = numpy.array(values)
        
        #Sort them first by index then by dr
        idx = numpy.lexsort(( points[:,1] , points[:,0]))
        points = points[idx]
        values = values[idx]
        
        #Now build one interpolator for each energy
        self.interpolators = []

        for i in range( self.eneGrid.shape[0] ):
    
            this = scipy.interpolate.LinearNDInterpolator(points, values[:,i])
    
            self.interpolators.append( this )
    
    def getTemplate( self, beta, DRbar):
        
        values = map( lambda interp: interp([beta, DRbar]), self.interpolators )
        
        return numpy.power(10, values)

class BandPPTemplate(SpectralModel):
    
    def setup(self):
        self.functionName       = "Band function multiplied by template"
        self.formula            = r'''
        \[f(E) = \left\{ \begin{eqnarray}
        K \left(\frac{E}{100 \mbox{ keV}}\right)^{\alpha} & \exp{\left(\frac{-E}{E_{c}}\right)} & \mbox{ if } E < (\alpha-\beta)E_{c} \\
        K \left[ (\alpha-\beta)\frac{E_{c}}{100}\right]^{\alpha-\beta}\left(\frac{E}{100 \mbox{ keV}}\right)^{\beta} & \exp{(\beta-\alpha)} & \mbox{ if } E \ge (\alpha-\beta)E_{c}
        \end{eqnarray}
        \right.
        \]
        '''

        self.parameters         = collections.OrderedDict()
        self.parameters['alpha'] = Parameter('alpha',-1.0,-5,10,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['beta']  = Parameter('beta',-2.0,-3,-1.5,0.1,fixed=False,nuisance=False,dataset=None)
        self.parameters['E0']    = Parameter('E0',500,10,1e5,10,fixed=False,nuisance=False,dataset=None,unit='keV')
        self.parameters['K']     = Parameter('K',1,1e-4,1e3,0.1,fixed=False,nuisance=False,dataset=None,normalization=True)
        
        self.parameters['Ec']    = Parameter('Ec',511,1e-4,1e6,100,fixed=False,nuisance=False,dataset=None,normalization=False)
        
        self.parameters['DRbar'] = Parameter('DRbar',1,0.0001,100,0.1,fixed=False,nuisance=False,dataset=None,normalization=False)
        
        def integral(e1,e2):
            return self((e1+e2)/2.0)*(e2-e1)
        self.integral            = integral
        
        self.interpolator = MyInterpolator()
        
        self.cache = {}
        
#        self.lookup = {}
#        
#        #Fill the lookup table with the already computed spectra
#        datfiles = glob.glob("flux_*.dat")
#        
#        for dat in datfiles:
#            
#            tokens = dat.split("_")
#            beta = float(tokens[2].replace("a",""))
#            DRbar = float(tokens[4].replace("DR","").replace(".dat",""))
#            
#            key = ("%.2f, %.2g" %(beta * (-1), DRbar))
#            
#            self.lookup[ key ] = dat
        
    
    def setTemplate(self, templateInstance):
        
        self.templateInstance = templateInstance
  
    def __call__(self,e):
        #The input e can be either a scalar or an array
        #The following will generate a wrapper which will
        #allow to treat them in exactly the same way,
        #as they both were arrays, while keeping the output
        #in line with the input: if e is a scalar, the output
        #will be a scalar; if e is an array, the output will be an array
        energies                 = numpy.array(e,ndmin=1,copy=False)
        alpha                    = self.parameters['alpha'].value
        beta                     = self.parameters['beta'].value
        E0                       = self.parameters['E0'].value
        K                        = self.parameters['K'].value
        Ec                       = self.parameters['Ec'].value
        
        DRbar                    = self.parameters['DRbar'].value
        
        if(alpha < beta):
          raise CustomExceptions.ModelAssertionViolation("Alpha cannot be less than beta")
        
        out                      = numpy.zeros(energies.flatten().shape[0])
        idx                      = (energies < (alpha-beta)*E0)
        nidx                     = ~idx
        out[idx]                 = numpy.maximum(K*numpy.power(energies[idx]/100.0,alpha)*numpy.exp(-energies[idx]/E0),1e-30)
        out[nidx]                = numpy.maximum(K*numpy.power((alpha-beta)*E0/100.0,alpha-beta)*numpy.exp(beta-alpha)*numpy.power(energies[nidx]/100.0,beta),1e-30)

        
#        key = ("%.2f, %.2g" %(beta, DRbar))
#        
#        if key in self.lookup.keys():
#            
#            print("Reusing beta = %.2f and DRbar = %.2g" %(beta, DRbar))
#            
#            template = self.lookup[key]
#        
#        else:
#            
#            print("Generating with beta = %.2f and DRbar = %.2g" %(beta, DRbar))
#            
#            #Generate template
#            template = fast_flux_computation.go( 0, 0, beta * (-1), DRbar, 1.0, 1.0, False )
#            
#            self.lookup[key] = template
#            
#        pass
#        
#        templIns = PairProduction( template, DRbar )
#        self.setTemplate( templIns )
#                
#        #Now multiply by the template
#        self.templateInstance.setCutoffEnergy(Ec)
        
        key = ("%.2f, %.2g" %(beta, DRbar))
        
        if key in self.cache.keys():
            
            cc = self.cache[key]
        
        else:
        
            nuFnu = self.interpolator.getTemplate( beta * (-1), DRbar )
        
            ee = self.interpolator.eneGrid * Ec
        
            #ene_interpolant = scipy.interpolate.UnivariateSpline(
            #                              numpy.log10( ee ), 
            #                              numpy.log10( nuFnu ), k=1,
            #                              s=0, ext=3)
        
            #cc = numpy.power(10, ene_interpolant(numpy.log10(energies)))
        
            interpolation = numpy.interp( numpy.log10( energies ), numpy.log10(ee), numpy.log10( nuFnu[:, 0] ) )
            cc = numpy.power(10, interpolation)
            
            self.cache[key] = cc
        
        pass
        
        #This should be out = (energies * energies * out * cc ) / energies / energies,
        #which of course simplify to:
        
        out                      = out * cc
        
        
        #This fixes nan(s) and inf values, converting them respectively to zeros and large numbers
        out                      = numpy.nan_to_num(out)
        
        if(out.shape[0]==1):
            return out[0]
        else:
            return out
 
